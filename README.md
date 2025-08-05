<p align="center">
  <img width="456" alt="Screenshot" src="https://github.com/user-attachments/assets/d1e71a06-6f25-42e0-ba5e-ec09250c392d" />
</p>

This is an [ISF shader](https://isf.video) for simulating the slime mold
_Physarum polycephalum_. The simulation is based on “Characteristics of pattern
formation and evolution in approximations of physarum transport networks” by
Jeff Jones (PDF available [here](https://uwe-repository.worktribe.com/output/980579)),
and [this webpage by Sage Johnson](https://cargocollective.com/sagejenson/physarum)
is another excellent resource. This particular shader is converted from
[this ShaderToy shader](https://www.shadertoy.com/view/ttsfWn) by
[**@MichaelMoroz**](https://github.com/MichaelMoroz).

This is a multi-pass shader that is intended to be used with floating-point
buffers. Not all ISF hosts support floating-point buffers.
[Videosync](https://videosync.showsync.com/download) supports floating-point
buffers in
[v2.0.12](https://support.showsync.com/release-notes/videosync/2.0#2012) and
later, but https://editor.isf.video does not appear to support floating-point
buffers. This shader will produce *very* different output if floating-point
buffers are not used.
