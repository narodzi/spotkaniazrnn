include("structures.jl")
import Base: sum

mat_mul(h :: GraphNode, W :: GraphNode) = BroadcastedOperator(mat_mul, h, W)
forward(::BroadcastedOperator{typeof(mat_mul)}, h, W) = return W * h
backward(::BroadcastedOperator{typeof(mat_mul)}, h, W, g) = return tuple(g * h', W' * g)

sum_op(Wh :: GraphNode, Ux :: GraphNode, b :: GraphNode) = BroadcastedOperator(sum_op, Wh, Ux, b)
forward(::BroadcastedOperator{typeof(sum_op)}, Wh, Ux, b) = return sum(Wh, Ux, b)
backward(::BroadcastedOperator{typeof(sum_op)}, Wh, Ux, b, g) = return tuple(ones(length(Wh)), ones(length(Ux)), ones(length(b)))

tan_h(x :: GraphNode) = BroadcastedOperator(tan_h, x)
forward(::BroadcastedOperator{typeof(tan_h)}, x) = return tanh.(x)
backward(::BroadcastedOperator{typeof(tan_h)}, x, g) = return g .* (1 .- tanh.(x) .^ 2)

dense(x::GraphNode, w::GraphNode) = BroadcastedOperator(dense, x, w)
forward(::BroadcastedOperator{typeof(dense)}, x, w) = w * x
backward(::BroadcastedOperator{typeof(dense)}, x, w, g) = tuple(w' * g, g * x', g)

identity(x::GraphNode) = BroadcastedOperator(identity, x)
forward(::BroadcastedOperator{typeof(identity)}, x) = x
backward(::BroadcastedOperator{typeof(identity)}, x, g) = tuple(g)

cross_entropy_loss(y_hat::GraphNode, y::GraphNode) = BroadcastedOperator(cross_entropy_loss, y_hat, y)
forward(::BroadcastedOperator{typeof(cross_entropy_loss)}, y_hat, y) = let
    global cumulative
    global correct_prediction

    cumulative += 1
    if argmax(y_hat) == argmax(y)
        correct_prediction += 1
    end
    
    y_hat = y_hat .- maximum(y_hat)
    y_hat = exp.(y_hat) ./ sum(exp.(y_hat))
    loss = sum(log.(y_hat) .* y) * -1.0
    return loss
end
backward(node::BroadcastedOperator{typeof(cross_entropy_loss)}, y_hat, y, g) = let
    y_hat = y_hat .- maximum(y_hat)
    y_hat = exp.(y_hat) ./ sum(exp.(y_hat))
    return tuple(g .* (y_hat .- y))
end