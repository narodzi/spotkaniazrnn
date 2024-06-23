using MLDatasets: MNIST
using Flux
include("model.jl")

function main()
    train_data = MNIST(split=:train)  
    test_data  = MNIST(split=:test)
    
    x_train = reshape(train_data.features, 28 * 28, :)
    y_train = Flux.onehotbatch(train_data.targets, 0:9)
    
    x_test = reshape(test_data.features, 28 * 28, :)
    y_test = Flux.onehotbatch(test_data.targets, 0:9)
    
    rnn = myRNN(
        Variable(Flux.glorot_uniform(64,64)), 
        Variable(Flux.glorot_uniform(64,14*14)),
        Variable(Flux.glorot_uniform(10,64)),
        Variable(zeros(Float32,64)),
        Variable(zeros(Float32, 10)),
        Variable(zeros(Float32, 64))
    )
    
    settings = (;
        eta = 15e-3,
        epochs = 5,
        batch_size = 100,
    )
    
    println("Training model...")
    train(rnn, x_train, y_train, settings.epochs, settings.batch_size, settings.eta)
    
    println("Testing model...")
    test(rnn, x_test, y_test)
end

main()