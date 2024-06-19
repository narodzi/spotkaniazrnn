abstract type GraphNode end
abstract type Operator <: GraphNode end

struct Constant{T} <: GraphNode
    output :: T
end

mutable struct Variable{T} <: GraphNode
    output :: T
    gradient :: Union{Nothing, T}
    name :: String    
    batch_gradient :: Union{Nothing, T}
    Variable(output::T; name::String = "?") where T = new{T}(output, nothing, name, nothing)
end

mutable struct BroadcastedOperator{F} <: Operator
    inputs :: Any
    output :: Any
    gradient :: Any
    name :: String
    BroadcastedOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
end