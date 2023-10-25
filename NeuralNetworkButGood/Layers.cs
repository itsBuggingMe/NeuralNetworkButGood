using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensornet;

namespace NeuralNetworkButGood
{
    internal interface ILayer
    {
        public int LayerSize { get; }
        public abstract Tensor<float> FeedForward(Tensor<float> WorkingVector);
        public ILayer CopyOf();
    }

    internal class GenericLayer : ILayer
    {
        public int LayerSize => _layerSize;
        private readonly int _layerSize;

        private readonly int _prevLayerSize;

        public Tensor<float> Weights;
        public Tensor<float> Biases;

        private Func<float, float> ActivationFunction;

        public GenericLayer(ILayer previousLayer, int layerSize, Func<float, float>? Activation = null)
        {
            this.Weights = Tensor.Random.Uniform<float>(new TensorShape(layerSize, previousLayer.LayerSize));
            this.Biases = Tensor.Random.Uniform<float>(new TensorShape(layerSize));

            this.ActivationFunction = Activation ?? ActivationFunctions.Relu;

            _layerSize = layerSize;
            _prevLayerSize = previousLayer.LayerSize;
        }

        private GenericLayer(int prevLayerSize, int layerSize, Func<float, float>? Activation = null)
        {
            this.Weights = Tensor.Random.Uniform<float>(new TensorShape(prevLayerSize, layerSize));
            this.Biases = Tensor.Random.Uniform<float>(new TensorShape(layerSize));

            this.ActivationFunction = Activation ?? ActivationFunctions.Relu;

            _prevLayerSize = prevLayerSize;
            _layerSize = layerSize;
        }

        public Tensor<float> FeedForward(Tensor<float> WorkingVector)
        {
            WorkingVector = (WorkingVector * Weights).Sum(1) + Biases;
            WorkingVector.ForEachInplace(ActivationFunction);
            return WorkingVector;
        }

        public ILayer CopyOf()
        {
            var copy = new GenericLayer(_prevLayerSize, LayerSize, ActivationFunction);
            Weights.CopyTo(copy.Weights);
            Biases.CopyTo(copy.Biases);
            return copy;
        }

        public void Mutate(float range, float prob)
        {
            Weights.ForEachInplace((v) =>
            {
                return GetOneRange(range, prob) + v;
            });

            Biases.ForEachInplace((v) =>
            {
                return GetOneRange(range, prob) + v;
            });
        }

        private float GetOneRange(float range, float prob)
        {
            if(Random.Shared.NextDouble() < prob)
            {
                return ((float)Random.Shared.NextDouble() - 0.5f) * range;
            }
            return 0;
        }

        internal static class ActivationFunctions
        {
            public static float Tanh(float value)
            {
                return MathF.Tanh(value);
            }

            public static float Relu(float value)
            {
                return Math.Max(value, 0);
            }

            public static float Sigmoid(float value)
            {
                return 1f / (1f + MathF.Exp(-value));
            }
        }
    }

    internal class InputLayer : ILayer
    {
        public int LayerSize => _layerSize;
        private readonly int _layerSize;

        public InputLayer(int layerSize)
        {
            _layerSize = layerSize;
        }

        public Tensor<float> FeedForward(Tensor<float> WorkingVector)
        {
            return WorkingVector;
        }

        public ILayer CopyOf()
        {
            return new InputLayer(_layerSize);
        }
    }
}
