/*{
    "CATEGORIES": [
        "Filter",
        "Generator"
    ],
    "CREDIT": "Mykhailo Moroz <https://www.shadertoy.com/user/michael0884>",
    "DESCRIPTION": "Random slime mold generator, converted from <https://www.shadertoy.com/view/ttsfWn>",
    "INPUTS": [
        {
            "NAME" : "inputImage",
            "TYPE" : "image"
        },
        {
            "NAME": "inputImageAmount",
            "LABEL": "Input image amount",
            "TYPE": "float",
            "DEFAULT": 0,
            "MIN": 0,
            "MAX": 1
        },
        {
            "NAME": "restart",
            "LABEL": "Restart",
            "TYPE": "event"
        },
        {
            "NAME": "dt",
            "LABEL": "Simulation speed",
            "TYPE": "float",
            "DEFAULT": 1,
            "MAX": 10,
            "MIN": 0
        },
        {
            "NAME": "distribution_size",
            "LABEL": "Trail size",
            "TYPE": "float",
            "DEFAULT": 1.2,
            "MAX": 10,
            "MIN": 0
        },
        {
            "NAME": "acceleration",
            "LABEL": "Particle acceleration",
            "TYPE": "float",
            "DEFAULT": 0.04,
            "MAX": 1,
            "MIN": 0
        },
        {
            "NAME": "sense_ang",
            "LABEL": "Sensor angle factor",
            "TYPE": "float",
            "DEFAULT": 1,
            "MAX": 2,
            "MIN": 0
        },
        {
            "NAME": "sense_dis",
            "LABEL": "Sensor distance",
            "TYPE": "float",
            "DEFAULT": 4,
            "MAX": 20,
            "MIN": 0
        },
        {
            "NAME": "distance_scale",
            "LABEL": "Sensor distance scale",
            "TYPE": "float",
            "DEFAULT": 2,
            "MAX": 1,
            "MIN": 0
        },
        {
            "NAME": "sense_oscil",
            "LABEL": "Sensor turn speed",
            "TYPE": "float",
            "DEFAULT": 0.2,
            "MAX": 1,
            "MIN": 0
        },
        {
            "NAME": "oscil_scale",
            "LABEL": "Sensor turn speed scale",
            "TYPE": "float",
            "DEFAULT": 0.5,
            "MAX": 1,
            "MIN": 0
        },
        {
            "NAME": "sense_force",
            "LABEL": "Sensor strength",
            "TYPE": "float",
            "DEFAULT": -0.01,
            "MAX": 1,
            "MIN": -1
        },
        {
            "NAME": "force_scale",
            "LABEL": "Sensor force scale",
            "TYPE": "float",
            "DEFAULT": 1.5,
            "MAX": 2,
            "MIN": 0
        }
    ],
    "ISFVSN": "2",
    "PASSES": [
        {
            "TARGET": "bufferA_positionAndMass",
            "PERSISTENT": true,
            "FLOAT": true
        },
        {
            "TARGET": "bufferA_velocity",
            "PERSISTENT": true,
            "FLOAT": true
        },
        {
            "TARGET": "bufferB",
            "PERSISTENT": true,
            "FLOAT": true
        },
        {
            "TARGET": "bufferC",
            "PERSISTENT": true,
            "FLOAT": true
        },
        {

        }
    ]
}
*/

//
// ShaderToy Common
//

#define PI 3.1415926535897932384626433832795

//mold stuff
#define sense_num 6

//SPH pressure
#define Pressure(rho) 0.5*rho
#define fluid_rho 0.2

//useful functions
#define GS(x) exp(-dot(x,x))
#define GS0(x) exp(-length(x))
#define Dir(ang) vec2(cos(ang), sin(ang))
#define Rot(ang) mat2(cos(ang), sin(ang), -sin(ang), cos(ang))

//data packing
#define POST_UNPACK(X) (clamp(X, 0., 1.) * 2. - 1.)
#define PRE_PACK(X) clamp(0.5 * X + 0.5, 0., 1.)


void main()
{
    vec2 position = gl_FragCoord.xy;

    if (PASSINDEX == 0 || PASSINDEX == 1) // ShaderToy Buffer A
    {
        vec2 X = vec2(0);
        vec2 V = vec2(0);
        float M = 0.;

        //basically integral over all updated neighbor distributions
        //that fall inside of this pixel
        //this makes the tracking conservative
        for (int i = -2; i <= 2; i++)
        for (int j = -2; j <= 2; j++) {
            vec2 translatedPosition = position + vec2(i, j);
            vec2 wrapped_tpos = mod(translatedPosition, RENDERSIZE);
            vec4 data = IMG_PIXEL(bufferA_positionAndMass, wrapped_tpos);

            vec2 X0 = POST_UNPACK(data.xy) + translatedPosition;
        	vec2 V0 = POST_UNPACK(IMG_PIXEL(bufferB, wrapped_tpos).xy);
        	float M0 = data.z;

            X0 += V0*dt; //integrate position

            //particle distribution size
            float K = distribution_size;

            vec4 aabbX = vec4(max(position - 0.5, X0 - K*0.5), min(position + 0.5, X0 + K*0.5)); //overlap aabb
            vec2 center = 0.5*(aabbX.xy + aabbX.zw); //center of mass
            vec2 size = max(aabbX.zw - aabbX.xy, 0.); //only positive

            //the deposited mass into this cell
            float m = M0*size.x*size.y/(K*K);

            //add weighted by mass
            X += center*m;
            V += V0*m;

            //add mass
            M += m;
        }

        //normalization
        if (M != 0.) {
            X /= M;
            V /= M;
        }

        //initial condition
        if (FRAMEINDEX < 1 || restart) {
            X = position;
            V = vec2(0.);
            M = 0.07*GS(-position/RENDERSIZE);
        }

        if (PASSINDEX == 0) {
            X = clamp(X - position, vec2(-0.5), vec2(0.5));
            gl_FragColor = vec4(PRE_PACK(X), M, 1.);
        } else {
            gl_FragColor = vec4(PRE_PACK(V), 0., 1.);
        }
    }
    else if (PASSINDEX == 2) // ShaderToy Buffer B
    {
        vec2 wrapped_pos = mod(position, RENDERSIZE);

        vec4 data = IMG_PIXEL(bufferA_positionAndMass, wrapped_pos);
        vec2 X = POST_UNPACK(data.xy) + position;
        vec2 V = POST_UNPACK(IMG_PIXEL(bufferA_velocity, wrapped_pos).xy);
        float M = data.z;

        if(M != 0.) //not vacuum
        {
            //Compute the SPH force
            vec2 F = vec2(0.);
            vec3 avgV = vec3(0.);

            for (int i = -2; i <= 2; i++)
            for (int j = -2; j <= 2; j++) {
                vec2 translatedPosition = position + vec2(i, j);
                vec2 wrapped_tpos = mod(translatedPosition, RENDERSIZE);
                vec4 data = IMG_PIXEL(bufferA_positionAndMass, wrapped_tpos);

                vec2 X0 = POST_UNPACK(data.xy) + translatedPosition;
                vec2 V0 = POST_UNPACK(IMG_PIXEL(bufferA_velocity, wrapped_tpos).xy);
                float M0 = data.z;
                vec2 dx = X0 - X;

                float avgP = 0.5*M0*(Pressure(M) + Pressure(M0));
                F -= 0.5*GS(1.*dx)*avgP*dx;
                avgV += M0*GS(1.*dx)*vec3(V0,1.);
            }
            avgV.xy /= avgV.z;

            float ang = atan(V.y, V.x);
            float dang = sense_ang*PI/float(sense_num);
            vec2 slimeF = vec2(0.);
            //slime mold sensors
            for (int i = -sense_num; i <= sense_num; i++) {
                float cang = ang + float(i) * dang;
            	vec2 dir = (1. + sense_dis*pow(M, distance_scale))*Dir(cang);
                vec2 sensedPosition = mod(X + dir, RENDERSIZE);
            	vec3 s0 = IMG_NORM_PIXEL(bufferC, sensedPosition / RENDERSIZE).xyz;
       			float fs = pow(s0.z, force_scale);
            	slimeF +=  sense_oscil*Rot(oscil_scale*(s0.z - M))*s0.xy
                         + sense_force*Dir(ang + sign(float(i))*PI*0.5)*fs;
            }

            //remove acceleration component and leave rotation
            slimeF -= dot(slimeF, normalize(V))*normalize(V);
            F += slimeF/float(2*sense_num);

            // if(iMouse.z > 0.)
            // {
            //     vec2 dx= position - iMouse.xy;
            //      F += 0.6*dx*GS(dx/20.);
            // }

            //integrate velocity
            V += F*dt/M;

            //acceleration for fun effects
            V *= 1. + acceleration;

            //velocity limit
            float v = length(V);
            V /= (v > 1.)?1.*v:1.;
        }

        //mass decay
       // M *= 0.999;

        //input
        //if(iMouse.z > 0.)
        //\\	M = mix(M, 0.5, GS((position - iMouse.xy)/13.));
        //else
         //   M = mix(M, 0.5, GS((position - RENDERSIZE*0.5)/13.));

        //save
        gl_FragColor = vec4(PRE_PACK(V), 0., 1.);
    }
    else if (PASSINDEX == 3) // ShaderToy Buffer C
    {
        float rho = 0.001;
        vec2 vel = vec2(0., 0.);

        //compute the smoothed density and velocity
        for (int i = -2; i <= 2; i++)
        for (int j = -2; j <= 2; j++) {
            vec2 translatedPosition = position + vec2(i, j);
            vec2 wrapped_tpos = mod(translatedPosition, RENDERSIZE);
            vec4 data = IMG_PIXEL(bufferA_positionAndMass, wrapped_tpos);

            vec2 X0 = POST_UNPACK(data.xy) + translatedPosition;
            vec2 V0 = POST_UNPACK(IMG_PIXEL(bufferB, wrapped_tpos).xy);
            float M0 = data.z;
            vec2 dx = X0 - position;

            #define radius 1.
            float K = GS(dx/radius)/(radius*radius);
            rho += M0*K;
            vel += M0*K*V0;
        }

        vel /= rho;

        gl_FragColor = vec4(vel, rho, 1.0);
    }
    else // ShaderToy Image
    {
        #ifdef heightmap
            // Normalized pixel coordinates
            position = (position - RENDERSIZE*0.5)/max(RENDERSIZE.x,RENDERSIZE.y);

            vec2 angles = vec2(0.5, -0.5)*PI;

            vec3 camera_z = vec3(cos(angles.x)*cos(angles.y),sin(angles.x)*cos(angles.y),sin(angles.y));
            vec3 camera_x = normalize(vec3(cos(angles.x+PI*0.5), sin(angles.x+PI*0.5),0.));
            vec3 camera_y = -normalize(cross(camera_x,camera_z));

            //tracking particle
            vec4 fp = vec4(RENDERSIZE*0.5 + 0.*vec2(150.*iTime, 0.), 0., 0.);

            vec3 ray = normalize(camera_z + FOV*(position.x*camera_x + position.y*camera_y));
            vec3 cam_pos = vec3(fp.xy-RENDERSIZE*0.5, 0.) - RAD*vec3(cos(angles.x)*cos(angles.y),sin(angles.x)*cos(angles.y),sin(angles.y));

            vec4 X = ray_march(cam_pos, ray);

            if(X.w < min_d)
            {

                float D = rho(X.xyz);
                vec3 albedo = vec3(1,0.3,0.3) + sin(1.*vec3(1.,0.2,0.1)*D);

                vec4 N0 = calcNormal(X.xyz, 2.*X.w)*vec4(5.,5.,1.,1.);
                vec3 n = normalize(N0.xyz);
                vec3 rd = reflect(ray, n);
                vec3 colA =texture(iChannel2,  rd.yzx).xyz;
                vec3 colB = 0.6*(vec3(0.5) + 0.5*dot(rd, normalize(vec3(1.))));
                colB += 3.*pow(max(dot(rd, normalize(vec3(1.))), 0.), 10.);
                colB += 3.*pow(max(dot(rd, normalize(vec3(-1,-0.5,0.8))), 0.), 10.);
                float b = clamp(0.5 + 0.5*dot(n, normalize(vec3(1,1,1))), 0.,1.);
                float K = 1. - pow(max(dot(n,rd),0.), 4.);
                gl_FragColor.rgb = 1.*albedo*colB + 0.3*colA*K;
            }
            else
            {
                //background
                gl_FragColor = 1.*texture(iChannel2,  ray.yzx);
            }
            gl_FragColor = tanh(1.3*gl_FragColor*gl_FragColor);
        #else
            vec2 wrapped_pos = mod(position, RENDERSIZE);
        	float r = IMG_NORM_PIXEL(bufferC, wrapped_pos / RENDERSIZE).z;

        	gl_FragColor.rgb = 3.*sin(0.2*vec3(1,2,3)*r);
        #endif

        gl_FragColor.a = 1.;
    }
}
