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
        },
        {
            "NAME": "massDecayFactor",
            "LABEL": "Mass decay",
            "TYPE": "float",
            "DEFAULT": 1,
            "MAX": 2,
            "MIN": 0
        },
        {
            "NAME": "radius",
            "LABEL": "Smoothing radius",
            "TYPE": "float",
            "DEFAULT": 1,
            "MAX": 10,
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

// Constants from LYGIA <https://github.com/patriciogonzalezvivo/lygia>
#define PI 3.1415926535897932384626433832795
#define HALF_PI 1.5707963267948966192313216916398

#define INV_SQRT_2 0.7071067811865475244008443621048

//
// ShaderToy Common
//

// Mold stuff
#define HALF_SENSOR_COUNT_MINUS_1 6

// Useful functions (from LYGIA <https://github.com/patriciogonzalezvivo/lygia>)
vec2 polar2cart(in vec2 polar) {
    return vec2(cos(polar.x), sin(polar.x)) * polar.y;
}

mat2 rotate2d(const in float r) {
    float c = cos(r);
    float s = sin(r);
    return mat2(c, s, -s, c);
}

float gaussian( vec2 d, float s) { return exp(-( d.x*d.x + d.y*d.y) / (2.0 * s*s)); }

// The ShaderToy shader uses the functions `floatBitsToUint` and
// `uintBitsToFloat` to pack more than 4 floats (5 in this case) into a
// 4-component pixel. These functions are available in GLSL v3.30 (OpenGL v3.3)
// and later, but some ISF hosts (notably Videosync) use GLSL v1.50
// (OpenGL v3.2). We can work around this by effectively running one of the
// ShaderToy buffers twice, but the packing operations in the ShaderToy shader
// also perform a `clamp` on the packed data. Without the `clamp` calls, this
// shader seems to blow up numerically.
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

        M += inputImageAmount * IMG_PIXEL(inputImage, position).x;

        // Mass decay
        M *= massDecayFactor;

        // Initial condition
        if (FRAMEINDEX < 1 || restart) {
            X = position;
            V = vec2(0);
            M = 0.07 * gaussian(-position / RENDERSIZE, INV_SQRT_2);
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

                float avgP = 0.5 * M0 * (0.5 * (M + M0));
                float positionChangeDistribution = gaussian(dx, INV_SQRT_2);
                F -= 0.5 * positionChangeDistribution * avgP * dx;
                avgV += M0 * positionChangeDistribution * vec3(V0, 1);
            }
            avgV.xy /= avgV.z;

            float ang = atan(V.y, V.x);
            float dang = sense_ang * PI / float(HALF_SENSOR_COUNT_MINUS_1);
            vec2 slimeF = vec2(0);
            // Slime mold sensors
            for (int i = -HALF_SENSOR_COUNT_MINUS_1; i <= HALF_SENSOR_COUNT_MINUS_1; i++) {
                float cang = ang + float(i) * dang;
            	vec2 dir = (1. + sense_dis * pow(M, distance_scale)) * polar2cart(vec2(cang, 1));
                vec2 sensedPosition = mod(X + dir, RENDERSIZE);
            	vec4 s0 = IMG_NORM_PIXEL(bufferC, sensedPosition / RENDERSIZE);
       			float fs = pow(s0.z, force_scale);
            	slimeF += sense_oscil * rotate2d(oscil_scale * (s0.z - M)) * s0.xy +
                          sense_force * polar2cart(vec2(ang + sign(float(i)) * HALF_PI, 1)) * fs;
            }

            // Remove acceleration component and leave rotation
            slimeF -= dot(slimeF, normalize(V)) * normalize(V);
            F += slimeF / float(2 * HALF_SENSOR_COUNT_MINUS_1);

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
            if (v > 1.) {
                V /= v;
            }
        }

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

            float K = gaussian(dx / radius, radius * INV_SQRT_2);
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

       	gl_FragColor = vec4(3. * sin(rho * 0.2 * vec3(1, 2, 3)), 1);
    }
}
