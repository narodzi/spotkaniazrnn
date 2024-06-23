include("structures.jl")
import Base: sum

rnnCell(U :: GraphNode, W :: GraphNode, h :: GraphNode, b :: GraphNode, x :: GraphNode) = BroadcastedOperator(rnnCell, U, W, h, b, x)
forward(::BroadcastedOperator{typeof(rnnCell)}, U, W, h, b, x) = tanh.(U * x .+ W * h .+ b)
backward(::BroadcastedOperator{typeof(rnnCell)}, U, W, h, b, x, g) = let 
    dh = g .* (1 .- tanh.((U * x) .+ (W * h) .+ b) .^ 2)
    return tuple(dh * x', dh * h', W' * dh, sum(dh, dims=2), U' * dh)
end

dense(x::GraphNode, w::GraphNode) = BroadcastedOperator(dense, x, w)
forward(::BroadcastedOperator{typeof(dense)}, x, w) = w * x
backward(::BroadcastedOperator{typeof(dense)}, x, w, g) = tuple(w' * g, g * x', g)

identity(x::GraphNode) = BroadcastedOperator(identity, x)
forward(::BroadcastedOperator{typeof(identity)}, x) = x
backward(::BroadcastedOperator{typeof(identity)}, x, g) = tuple(g)

cross_entropy_loss(y_hat::GraphNode, y::GraphNode) = BroadcastedOperator(cross_entropy_loss, y_hat, y)
forward(::BroadcastedOperator{typeof(cross_entropy_loss)}, y_hat, y) = let
    global predictions
    global correct_predictions

    predictions += 1
    if argmax(y_hat) == argmax(y)
        correct_predictions += 1
    end
    
    y_h = exp.(y_hat .- maximum(y_hat)) ./ sum(exp.(y_hat .- maximum(y_hat)))
    loss = -sum(log.(y_h) .* y)
    return loss
end
backward(::BroadcastedOperator{typeof(cross_entropy_loss)}, y_hat, y, g) = let
    y_h = exp.(y_hat .- maximum(y_hat)) ./ sum(exp.(y_hat .- maximum(y_hat)))
    return tuple(g .* (y_h .- y))
end