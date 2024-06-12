using Random
using Printf
include("structures.jl")
include("backward-pass.jl")
include("forward-pass.jl")
include("operators.jl")
include("graph-builder.jl")

correct_prediction = 0
cumulative = 0

struct myRNN
    WW#Matrix{Float64}
    WU#Matrix{Float64}
    WV#Matrix{Float64}
    bh#Vector{Float64}
    by#Vector{Float64}
    h#Vector{Float64}
end

function update_weights!(graph::Vector, learning_rate::Float64, batch_size::Int64)
    for node in graph
        if isa(node, Variable) && hasproperty(node, :batch_gradient)
            node.batch_gradient ./= batch_size
            node.output .-= learning_rate * node.batch_gradient
            fill!(node.batch_gradient, 0)
        end
    end
end

function build_graph(x::Constant, y::Constant, rnn::myRNN)
    # l1 = mat_mul(rnn.WW, rnn.h)
    # l2 = mat_mul(rnn.WU, x)
    # l3 = sum_op(l1, l2, rnn.bh)
    # l4 = tan_h(l3)
    l1 = rnnCell(rnn.WU, rnn.WW, rnn.h, rnn.bh, x)
    l5 = dense(l1, rnn.WV)
    e = cross_entropy_loss(l5, y)

	return topological_sort(e)
end


function train(rnn::myRNN, x::Any, y::Any, epochs, batch_size, learining_rate)

    @time for i=1:epochs

        epoch_loss = 0.0
        samples = size(x, 2)

        global correct_prediction = 0
        global cumulative = 0

        println("Epoch: ", i)

        for k=1:196:size(x,1)
            for j=1:samples        


                x_train = Constant(x[k:k+195, j])
                y_train = Constant(y[:, j])

                graph = build_graph(x_train, y_train, rnn)
                epoch_loss += forward!(graph)
                backward!(graph)

                if j % batch_size == 0
                    update_weights!(graph, learining_rate, batch_size)
                end
            end
        end 

        @printf("   Average loss: %.4f\n", epoch_loss/samples)
        @printf("   Train accuracy: %.4f\n", correct_prediction/cumulative)

    end
end


function test(rnn::myRNN, x::Any, y::Any)

    samples = size(x, 3)

    global correct_prediction
    global cumulative

    for i=1:samples

        x_train = Constant(x[:, :, i])
        y_train = Constant(y[i, :])

        graph = build_graph(x_train, y_train, rnn)
		forward!(graph)

    end

    @printf("Test accuracy: %.4f\n\n", correct_prediction/cumulative)
end