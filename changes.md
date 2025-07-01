
In order ot update to a newer Julia versions, I made the following changes:
- remove old ONNX reader (main reason for restricting newer versions) and replace it with `VNNLib.jl`
    - `VNNLib.jl` also has an ONNX reader, but it can **only read** ONNX, not write it. It also doesn't really convert the ONNX models to Flux (some ONNX operations just require some more manipulation), but transforms each node to a respective Julia type that can be executed using Flux *and additional Julia functions*.
- don't call `import Flux: flatten` because it collides with the same function from `ReachabilityAnalysis`, just use the prefixed version `Flux.flatten` instead
- add `src/LazySetsAdapter/RotatedHyperrectangle.jl` because `RotatedHyperrectangle` was removed from `LazySets` in version `3.0.0`.
- removed the constructor `BetaCrown(nothing)` because it was preventing precompilation.
    - the constructor `BetaCrown(nothing; use_gpu=false)` is still there (thus the constructor above should not have been necessary in the first place)
- change from `Flux.@functor` to `Flux.@layer` as recommended for Flux `0.15` and higher