using MLDatasets: MNIST
using Flux
include("model.jl")

train_data = MNIST(split=:train)  
test_data  = MNIST(split=:test)

x_train = reshape(train_data.features, 28 * 28, :)
y_train  = Flux.onehotbatch(train_data.targets, 0:9)

x_test = reshape(test_data.features, 28 * 28, :)
y_test  = Flux.onehotbatch(test_data.targets, 0:9)

WW = Variable(Flux.glorot_uniform(64,64))
WU = Variable(Flux.glorot_uniform(64,14*14))
WV = Variable(Flux.glorot_uniform(10,64))
bh = Variable(zeros(64))
by = Variable(zeros(10))
h = Variable(zeros(64))

rnn = myRNN(WW, WU, WV, bh, by, h)

settings = (;
    eta = 15e-3,
    epochs = 5,
    batch_size = 100,
)

println("Training model...")
train(rnn, x_train, y_train, settings.epochs, settings.batch_size, settings.eta)

println("Testing model...")
test(rnn, x_test, y_test) 