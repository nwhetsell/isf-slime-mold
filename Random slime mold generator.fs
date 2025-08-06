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

        // Basically integrate over all updated neighbor distributions that fall
        // inside of this pixel. This makes the tracking conservative.
        for (int i = -2; i <= 2; i++)
        for (int j = -2; j <= 2; j++) {
            vec2 translatedPosition = position + vec2(i, j);
            vec2 wrappedPosition = mod(translatedPosition, RENDERSIZE);
            vec4 data = IMG_PIXEL(bufferA_positionAndMass, wrappedPosition);

            vec2 X0 = POST_UNPACK(data.xy) + translatedPosition;
        	vec2 V0 = POST_UNPACK(IMG_PIXEL(bufferB, wrappedPosition).xy);
        	float M0 = data.z;

            X0 += V0 * dt; // Integrate position

            // Overlap aabb
            vec4 aabbX = vec4(
                max(position - 0.5, X0 - 0.5 * distribution_size),
                min(position + 0.5, X0 + 0.5 * distribution_size)
            );
            vec2 center = 0.5 * (aabbX.xy + aabbX.zw); // Center of mass
            vec2 size = max(aabbX.zw - aabbX.xy, 0.); // Only positive

            // Deposited mass into this cell
            float m = M0 * size.x * size.y / (distribution_size * distribution_size);

            // Add weighted by mass
            X += center * m;
            V += V0 * m;

            // Add mass
            M += m;
        }

        // Normalization
        if (M != 0.) {
            X /= M;
            V /= M;
        }

        // Initial condition
        if (FRAMEINDEX < 1 || restart) {
            X = position;
            V = vec2(0);
            M = 0.07 * GS(-position / RENDERSIZE);
        }

        if (PASSINDEX == 0) {
            X = clamp(X - position, vec2(-0.5), vec2(0.5));
            gl_FragColor = vec4(PRE_PACK(X), M, 1);
        } else {
            gl_FragColor = vec4(PRE_PACK(V), 0, 1);
        }
    }
    else if (PASSINDEX == 2) // ShaderToy Buffer B
    {
        vec2 wrappedPosition = mod(position, RENDERSIZE);

        vec4 data = IMG_PIXEL(bufferA_positionAndMass, wrappedPosition);
        vec2 X = POST_UNPACK(data.xy) + position;
        vec2 V = POST_UNPACK(IMG_PIXEL(bufferA_velocity, wrappedPosition).xy);
        float M = data.z;

        if (M != 0.) { // Not vacuum
            // Compute the SPH force
            vec2 F = vec2(0);
            vec3 avgV = vec3(0);

            for (int i = -2; i <= 2; i++)
            for (int j = -2; j <= 2; j++) {
                vec2 translatedPosition = position + vec2(i, j);
                wrappedPosition = mod(translatedPosition, RENDERSIZE);
                vec4 data = IMG_PIXEL(bufferA_positionAndMass, wrappedPosition);

                vec2 X0 = POST_UNPACK(data.xy) + translatedPosition;
                vec2 V0 = POST_UNPACK(IMG_PIXEL(bufferA_velocity, wrappedPosition).xy);
                float M0 = data.z;
                vec2 dx = X0 - X;

                float avgP = 0.5 * M0 * (Pressure(M) + Pressure(M0));
                F -= 0.5 * GS(dx) * avgP * dx;
                avgV += M0 * GS(dx) * vec3(V0, 1);
            }
            avgV.xy /= avgV.z;

            float ang = atan(V.y, V.x);
            float dang = sense_ang * PI / float(sense_num);
            vec2 slimeF = vec2(0);
            // Slime mold sensors
            for (int i = -sense_num; i <= sense_num; i++) {
                float cang = ang + float(i) * dang;
            	vec2 dir = (1. + sense_dis * pow(M, distance_scale)) * Dir(cang);
                vec2 sensedPosition = mod(X + dir, RENDERSIZE);
            	vec4 s0 = IMG_NORM_PIXEL(bufferC, sensedPosition / RENDERSIZE);
       			float fs = pow(s0.z, force_scale);
            	slimeF += sense_oscil * Rot(oscil_scale*(s0.z - M)) * s0.xy +
                          sense_force * Dir(ang + sign(float(i)) * 0.5 * PI) * fs;
            }

            // Remove acceleration component and leave rotation
            slimeF -= dot(slimeF, normalize(V)) * normalize(V);
            F += slimeF / float(2 * sense_num);

            // if (iMouse.z > 0.) {
            //     vec2 dx = position - iMouse.xy;
            //     F += 0.6 *dx * GS(dx / 20.);
            // }

            // Integrate velocity
            V += F * dt / M;

            // Acceleration for fun effects
            V *= 1. + acceleration;

            // Velocity limit
            float v = length(V);
            V /= (v > 1.) ? v : 1.;
        }

        // // Mass decay
        // M *= 0.999;

        // // Input
        // if (iMouse.z > 0.) {
        //     M = mix(M, 0.5, GS((position - iMouse.xy) / 13.));
        // } else {
        //     M = mix(M, 0.5, GS((position - 0.5 * RENDERSIZE) / 13.));
        // }

        gl_FragColor = vec4(PRE_PACK(V), 0, 1);
    }
    else if (PASSINDEX == 3) // ShaderToy Buffer C
    {
        float rho = 0.001;
        vec2 vel = vec2(0);

        // Compute the smoothed density and velocity
        for (int i = -2; i <= 2; i++)
        for (int j = -2; j <= 2; j++) {
            vec2 translatedPosition = position + vec2(i, j);
            vec2 wrappedPosition = mod(translatedPosition, RENDERSIZE);
            vec4 data = IMG_PIXEL(bufferA_positionAndMass, wrappedPosition);

            vec2 X0 = POST_UNPACK(data.xy) + translatedPosition;
            vec2 V0 = POST_UNPACK(IMG_PIXEL(bufferB, wrappedPosition).xy);
            float M0 = data.z;
            vec2 dx = X0 - position;

            #define radius 1.
            float K = GS(dx / radius ) / (radius * radius);
            rho += M0 * K;
            vel += M0 * K * V0;
        }

        vel /= rho;

        gl_FragColor = vec4(vel, rho, 1);
    }
    else // ShaderToy Image
    {
        vec2 wrappedPosition = mod(position, RENDERSIZE);
       	float rho = IMG_NORM_PIXEL(bufferC, wrappedPosition / RENDERSIZE).z;

       	gl_FragColor.rgb = 3. * sin(rho * 0.2 * vec3(1, 2, 3));
        gl_FragColor.a = 1.;
    }
}
