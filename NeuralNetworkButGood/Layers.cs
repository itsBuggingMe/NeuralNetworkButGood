using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensornet;

namespace NeuralNetworkButGood
{
    public interface ILayer
    {
        public int LayerSize { get; }
        public abstract Tensor<float> FeedForward(Tensor<float> WorkingVector);
    }

    public interface IWeightable
    {
        public Tensor<float> Weights { get; }
    }
    public interface IBiasable
    {
        public Tensor<float> Biases { get; }
    }
    public interface IActivation
    {
        public Func<float, float> ActivationFunction { get; }
    }

    public class GenericLayer : ILayer, IWeightable, IBiasable, IActivation
    {
        public int LayerSize => _layerSize;
        private readonly int _layerSize;


        public Tensor<float> Weights => _weights;
        private Tensor<float> _weights;

        public Tensor<float> Biases => _biases;
        private Tensor<float> _biases;

        public Func<float, float> ActivationFunction => _activationFunction;
        private Func<float, float> _activationFunction;

        /// <summary>
        /// Default Activation is Relu
        /// </summary>
        public GenericLayer(int previousLayerSize, int layerSize, Func<float, float>? Activation = null)
        {
            this._weights = Tensor.Zeros<float>(new TensorShape(layerSize, previousLayerSize));
            this._biases = Tensor.Zeros<float>(new TensorShape(layerSize));

            Weights.ForEachInplace((f) => { return (float)NetworkUtils.GenerateRandom.NextDouble() * 2 - 1; });
            Biases.ForEachInplace((f) => { return (float)NetworkUtils.GenerateRandom.NextDouble() * 2 - 1; });
            


            _activationFunction = Activation ?? Activations.Relu;

            _layerSize = layerSize;
        }

        public Tensor<float> FeedForward(Tensor<float> WorkingVector)
        {
            WorkingVector = (WorkingVector * Weights).Sum(1) + Biases;
            WorkingVector.ForEachInplace(ActivationFunction);
            return WorkingVector;
        }
    }

    public class SoftMaxFullConnected : ILayer, IWeightable, IBiasable
    {
        public int LayerSize => _layerSize;
        private readonly int _layerSize;

        public Tensor<float> Weights => _weights;
        private Tensor<float> _weights;

        public Tensor<float> Biases => _biases;
        private Tensor<float> _biases;

        public SoftMaxFullConnected(int previousLayer, int layerSize)
        {
            this._weights = Tensor.Zeros<float>(new TensorShape(layerSize, previousLayer));
            this._biases = Tensor.Zeros<float>(new TensorShape(layerSize));

            Weights.ForEachInplace((f) => { return (float)NetworkUtils.GenerateRandom.NextDouble() * 2 - 1; });
            Biases.ForEachInplace((f) => { return (float)NetworkUtils.GenerateRandom.NextDouble() * 2 - 1; });

            _layerSize = layerSize;
        }

        public Tensor<float> FeedForward(Tensor<float> WorkingVector)
        {
            WorkingVector = ((WorkingVector * _weights).Sum(1) + _biases);

            WorkingVector.ForEachInplace((f) =>
            {
                return MathF.Pow(MathF.E, f);
            });

            float sum = WorkingVector.Sum()[0];
            float inverseSum = 1f / sum;
            return WorkingVector * inverseSum;
        }
    }

    public class InputLayer : ILayer
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
    }

    public static class Activations
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
