This is an [ISF shader](https://isf.video) for simulating the slime mold
_Physarum polycephalum_. The simulation is based on “Characteristics of pattern
formation and evolution in approximations of physarum transport networks” by
Jeff Jones (PDF available [here](https://uwe-repository.worktribe.com/output/980579)),
and [this webpage by Sage Johnson](https://cargocollective.com/sagejenson/physarum)
is another excellent resource. This particular shader is converted from
[this ShaderToy shader](https://www.shadertoy.com/view/ttsfWn) by
[**@MichaelMoroz**](https://github.com/MichaelMoroz).

This shader relies on data packing with
[`floatBitsToUint`](https://registry.khronos.org/OpenGL-Refpages/gl4/html/floatBitsToInt.xhtml)
and
[`uintBitsToFloat`](https://registry.khronos.org/OpenGL-Refpages/gl4/html/intBitsToFloat.xhtml),
which are available in GLSL v3.30
([released in 2010](https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.3.30.pdf))
and later. Some ISF hosts (like
[Videosync](https://videosync.showsync.com/download)) use GLSL v1.5 or earlier,
and so they cannot run this shader. Modern versions of
[Cycling ’74 Max](https://cycling74.com/products/max) (probably
[v8.5 and later](https://docs.cycling74.com/userguide/jitter/graphics_engine/))
use a version of GLSL that includes `floatBitsToUint` and `uintBitsToFloat`, so
the jit.gl.isf object can render this shader.
