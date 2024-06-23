using Random
using Printf
include("structures.jl")
include("backward-pass.jl")
include("forward-pass.jl")
include("operators.jl")
include("graph-builder.jl")

predictions = 0
correct_predictions = 0

struct myRNN
    WW :: Variable{Matrix{Float32}}
    WU :: Variable{Matrix{Float32}}
    WV :: Variable{Matrix{Float32}}
    bh :: Variable{Vector{Float32}}
    by :: Variable{Vector{Float32}}
    h :: Variable{Vector{Float32}}
end

function update_weights!(graph::Vector, lr::Float64, batch_size::Int64)
    for node in graph
        if isa(node, Variable)
			node.batch_gradient ./= batch_size
            node.output .-= lr * node.batch_gradient 
            node.batch_gradient .= 0
        end
    end
end


function build_graph(x::Matrix{Float32}, y, rnn::myRNN, j:: Number)
    l1 = rnnCell(rnn.WU, rnn.WW, rnn.h, rnn.bh, Constant(x[1:196, j]))
    l2 = rnnCell(rnn.WU, rnn.WW, l1, rnn.bh, Constant(x[197:392, j]))
    l3 = rnnCell(rnn.WU, rnn.WW, l2, rnn.bh, Constant(x[393:588, j]))
    l4 = rnnCell(rnn.WU, rnn.WW, l3, rnn.bh, Constant(x[589:end, j]))
    l5 = dense(l4, rnn.WV, rnn.by) |> identity
    e = cross_entropy_loss(l5, Constant(y[:, j]))

    return topological_sort(e)
end

function train(rnn::myRNN, x::Matrix{Float32},  y, epochs:: Int64, batch_size:: Int64, learning_rate:: Float64)

    @time for i=1:epochs

        epoch_loss = 0.0
        samples = size(x, 2)

        global correct_predictions = 0
        global predictions = 0

        @time for j=1:samples            
            graph = build_graph(x, y, rnn, j)
            epoch_loss += forward!(graph)
            backward!(graph)

            if j % batch_size == 0
                update_weights!(graph, learning_rate, batch_size)
            end
        end

        epoch = "Epoch $i"
        loss = epoch_loss/samples
        acc_calc = round(100 * (correct_predictions/predictions), digits=2)
        train_acc = "$acc_calc %"

        @info epoch loss train_acc
    end
end


function test(rnn::myRNN, x::Matrix{Float32}, y)

    samples = size(x, 2)

    global correct_predictions = 0
    global predictions = 0

    epoch_loss = 0.0

    @time for j=1:samples
        graph = build_graph(x, y, rnn, j)
        epoch_loss += forward!(graph)
    end

    test = "Test"
    acc_calc = round(100 * (correct_predictions/predictions), digits=2)
    test_acc = "$acc_calc %"
    loss = epoch_loss/samples

    @info loss test test_acc
end