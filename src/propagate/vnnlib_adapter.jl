
"""
This file contains code to allow ModelVerification to work with networks parsed using VNNLib.ONNXParser.

The file currently only provides an adapter with functions dispatching on the type of the ONNXParser node and 
then directly calling the ModelVerification's code for the corresponding Flux layer.
"""


function propagate_layer_batch(prop_method, node::OXP.ONNXLinear, bound, batch_info)
    # we don't currently support transpose, because we directly call the code for the Flux dense layer 
    # provided in ModelVerification.
    # To handle transpose, we would need to also call code for transposing!
    @assert node.transpose == false "Transpose is currently not supported for ONNXLinear!"
    propagate_layer_batch(prop_method, node.dense, bound, batch_info)
end


function propagate_layer_batch(prop_method, node::OXP.ONNXRelu, bound, batch_info)
    propagate_layer_batch(prop_method, Flux.relu, bound, batch_info)
end