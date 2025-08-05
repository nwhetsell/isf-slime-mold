/*{
    "CATEGORIES": [
        "Generator"
    ],
    "CREDIT": "Mykhailo Moroz <https://www.shadertoy.com/user/michael0884>",
    "DESCRIPTION": "Random slime mold generator, converted from <https://www.shadertoy.com/view/ttsfWn>",
    "INPUTS": [
        {
            "NAME": "restart",
            "LABEL": "Restart",
            "TYPE": "event"
        },
        {
            "NAME": "iMouse",
            "TYPE": "point2D",
            "DEFAULT": [0.5, 0.5],
            "MIN": [0, 0],
            "MAX": [1, 1]
        }
    ],
    "ISFVSN": "2",
    "PASSES": [
        {
            "TARGET": "bufferA",
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
#define dt 1.
#define R iResolution.xy


// Hash function from <https://www.shadertoy.com/view/4djSRW>, MIT-licensed:
//
// Copyright © 2014 David Hoskins.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the “Software”), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
float hash11(float p)
{
    p = fract(p * 0.1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}

#define rand_interval 250
#define random_gen(a, b, seed) ((a) + ((b)-(a))*hash11(seed + float(iFrame/rand_interval)))


#define distribution_size 1.2

//mold stuff
#define sense_num 6
#define sense_ang random_gen(0.1, 1., 40.)
#define sense_dis random_gen(4., 20., 10.)
#define sense_oscil random_gen(0., 0.2, 20.)
#define oscil_scale 0.5
#define sense_force random_gen(-0.15, 0.05, 30.)
#define distance_scale random_gen(0., 1.0, 70.)
#define force_scale 1.5
#define trailing 0.
#define acceleration random_gen(0., 0.08, 50.)

//SPH pressure
#define Pressure(rho) 0.5*rho
#define fluid_rho 0.2

//useful functions
#define GS(x) exp(-dot(x,x))
#define GS0(x) exp(-length(x))
#define Dir(ang) vec2(cos(ang), sin(ang))
#define Rot(ang) mat2(cos(ang), sin(ang), -sin(ang), cos(ang))
#define loop(i,x) for(int i = 0; i < x; i++)
#define range(i,a,b) for(int i = a; i <= b; i++)

//data packing
#define PACK(X) ( uint(round(65534.0*clamp(0.5*X.x+0.5, 0., 1.))) + \
           65535u*uint(round(65534.0*clamp(0.5*X.y+0.5, 0., 1.))) )

#define UNPACK(X) (clamp(vec2(X%65535u, X/65535u)/65534.0, 0.,1.)*2.0 - 1.0)

#ifdef VIDEOSYNC
uint floatBitsToUint(float x);
float uintBitsToFloat(uint x);
#endif

#define DECODE(X) UNPACK(floatBitsToUint(X))
#define ENCODE(X) uintBitsToFloat(PACK(X))


#define pos gl_FragCoord.xy
#define iFrame FRAMEINDEX
#define iResolution RENDERSIZE
#define U gl_FragColor
#define fragColor gl_FragColor
#define col gl_FragColor

void main()
{
    if (PASSINDEX == 0) // ShaderToy Buffer A
    {
        ivec2 p = ivec2(pos);

        vec2 X = vec2(0);
        vec2 V = vec2(0);
        float M = 0.;

        //basically integral over all updated neighbor distributions
        //that fall inside of this pixel
        //this makes the tracking conservative
        range(i, -2, 2) range(j, -2, 2)
        {
            vec2 tpos = pos + vec2(i,j);
            vec4 data = texelFetch(bufferB, ivec2(mod(tpos, R)), 0);

            vec2 X0 = DECODE(data.x) + tpos;
        	vec2 V0 = DECODE(data.y);
        	vec2 M0 = data.zw;

            X0 += V0*dt; //integrate position

            //particle distribution size
            float K = distribution_size;

            vec4 aabbX = vec4(max(pos - 0.5, X0 - K*0.5), min(pos + 0.5, X0 + K*0.5)); //overlap aabb
            vec2 center = 0.5*(aabbX.xy + aabbX.zw); //center of mass
            vec2 size = max(aabbX.zw - aabbX.xy, 0.); //only positive

            //the deposited mass into this cell
            float m = M0.x*size.x*size.y/(K*K);

            //add weighted by mass
            X += center*m;
            V += V0*m;

            //add mass
            M += m;
        }

        //normalization
        if(M != 0.)
        {
            X /= M;
            V /= M;
        }

        //initial condition
        if(iFrame < 1 || restart)
        {
            X = pos;
            V = vec2(0.);
            M = 0.07*GS(-pos/R);
        }

        X = clamp(X - pos, vec2(-0.5), vec2(0.5));
        U = vec4(ENCODE(X), ENCODE(V), M, 0.);
    }
    else if (PASSINDEX == 1) // ShaderToy Buffer B
    {
        vec2 uv = pos/R;
        ivec2 p = ivec2(pos);

        vec4 data = texelFetch(bufferA, ivec2(mod(pos, R)), 0);
        vec2 X = DECODE(data.x) + pos;
        vec2 V = DECODE(data.y);
        float M = data.z;

        if(M != 0.) //not vacuum
        {
            //Compute the SPH force
            vec2 F = vec2(0.);
            vec3 avgV = vec3(0.);
            range(i, -2, 2) range(j, -2, 2)
            {
                vec2 tpos = pos + vec2(i,j);
                vec4 data = texelFetch(bufferA, ivec2(mod(tpos, R)), 0);

                vec2 X0 = DECODE(data.x) + tpos;
                vec2 V0 = DECODE(data.y);
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
            range(i, -sense_num, sense_num)
            {
                float cang = ang + float(i) * dang;
            	vec2 dir = (1. + sense_dis*pow(M, distance_scale))*Dir(cang);
            	vec3 s0 = texture(bufferC, mod(X + dir, R) / R).xyz;
       			float fs = pow(s0.z, force_scale);
            	slimeF +=  sense_oscil*Rot(oscil_scale*(s0.z - M))*s0.xy
                         + sense_force*Dir(ang + sign(float(i))*PI*0.5)*fs;
            }

            //remove acceleration component and leave rotation
            slimeF -= dot(slimeF, normalize(V))*normalize(V);
            F += slimeF/float(2*sense_num);

            // if(iMouse.z > 0.)
            // {
            //     vec2 dx= pos - iMouse.xy;
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
        //\\	M = mix(M, 0.5, GS((pos - iMouse.xy)/13.));
        //else
         //   M = mix(M, 0.5, GS((pos - R*0.5)/13.));

        //save
        X = clamp(X - pos, vec2(-0.5), vec2(0.5));
        U = vec4(ENCODE(X), ENCODE(V), M, 0.);
    }
    else if (PASSINDEX == 2) // ShaderToy Buffer C
    {
        float rho = 0.001;
        vec2 vel = vec2(0., 0.);

        //compute the smoothed density and velocity
        range(i, -2, 2) range(j, -2, 2)
        {
            vec2 tpos = pos + vec2(i,j);
            vec4 data = texelFetch(bufferB, ivec2(mod(tpos, R)), 0);

            vec2 X0 = DECODE(data.x) + tpos;
            vec2 V0 = DECODE(data.y);
            float M0 = data.z;
            vec2 dx = X0 - pos;

            #define radius 1.
            float K = GS(dx/radius)/(radius*radius);
            rho += M0*K;
            vel += M0*K*V0;
        }

        vel /= rho;

        fragColor = vec4(vel, rho, 1.0);
    }
    else // ShaderToy Image
    {
        #ifdef heightmap
            // Normalized pixel coordinates
            pos = (pos - R*0.5)/max(R.x,R.y);

            vec2 uv = iMouse.xy/R;
            vec2 angles = vec2(0.5, -0.5)*PI;

            vec3 camera_z = vec3(cos(angles.x)*cos(angles.y),sin(angles.x)*cos(angles.y),sin(angles.y));
            vec3 camera_x = normalize(vec3(cos(angles.x+PI*0.5), sin(angles.x+PI*0.5),0.));
            vec3 camera_y = -normalize(cross(camera_x,camera_z));

            //tracking particle
            vec4 fp = vec4(R*0.5 + 0.*vec2(150.*iTime, 0.), 0., 0.);

            vec3 ray = normalize(camera_z + FOV*(pos.x*camera_x + pos.y*camera_y));
            vec3 cam_pos = vec3(fp.xy-R*0.5, 0.) - RAD*vec3(cos(angles.x)*cos(angles.y),sin(angles.x)*cos(angles.y),sin(angles.y));

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
                col.xyz = 1.*albedo*colB + 0.3*colA*K;
            }
            else
            {
                //background
                col = 1.*texture(iChannel2,  ray.yzx);
            }
            col = tanh(1.3*col*col);
        #else
        	float r = texture(bufferC, mod(pos.xy, R) / R).z;

        	col.xyz =  3.*sin(0.2*vec3(1,2,3)*r);
        #endif

        col.a = 1.;
    }
}
