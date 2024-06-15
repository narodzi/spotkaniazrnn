using Random
using Printf
include("structures.jl")
include("backward-pass.jl")
include("forward-pass.jl")
include("operators.jl")
include("graph-builder.jl")

correct_prediction = 0
cumulative = 0

mutable struct myRNN
    WW#Matrix{Float64}
    WU#Matrix{Float64}
    WV#Matrix{Float64}
    bh#Vector{Float64}
    by#Vector{Float64}
    h#Vector{Float64}
end

function update_weights!(graph::Vector, lr::Float64, batch_size::Int64)
    for node in graph
        if isa(node, Variable) && hasproperty(node, :batch_gradient)
			node.batch_gradient ./= batch_size
            node.output .-= lr * node.batch_gradient 
            fill(node.batch_gradient, 0)
        end
    end
end


function build_graph(x, y, rnn::myRNN, j:: Number)
    l1 = rnnCell(rnn.WU, rnn.WW, rnn.h, rnn.bh, Constant(x[1:196, j]))
    l2 = rnnCell(rnn.WU, rnn.WW, l1, rnn.bh, Constant(x[197:392, j]))
    l3 = rnnCell(rnn.WU, rnn.WW, l2, rnn.bh, Constant(x[393:588, j]))
    l4 = rnnCell(rnn.WU, rnn.WW, l3, rnn.bh, Constant(x[589:end, j]))
    l5 = dense(l4, rnn.WV) |> identity
    e = cross_entropy_loss(l5, y)

    return topological_sort(e)
end


function train(rnn::myRNN, x::Any, y::Any, epochs, batch_size, learning_rate)

    @time for i=1:epochs

        epoch_loss = 0.0
        samples = size(x, 2)

        global correct_prediction = 0
        global cumulative = 0

        println("Epoch: ", i)

        for j=1:samples
            y_train = Constant(y[:, j])
            
            graph = build_graph(x, y_train, rnn, j)
            rnn.h = Variable(zeros(64))
            epoch_loss += forward!(graph)
            backward!(graph)

            if j % batch_size == 0
                update_weights!(graph, learning_rate, batch_size)
            end
        end

        @printf("   Average loss: %.4f\n", epoch_loss/samples)
        @printf("   Train accuracy: %.4f\n", correct_prediction/cumulative)

    end
end


function test(rnn::myRNN, x::Any, y::Any)

    samples = size(x, 2)

    global correct_prediction
    global cumulative

    for j=1:samples
        y_train = Constant(y[:, j])
        graph = build_graph(x, y_train, rnn, j)
        rnn.h = Variable(zeros(64))
        forward!(graph)
    end

    @printf("Test accuracy: %.4f\n\n", correct_prediction/cumulative)
end