using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.CompilerServices;
using Tensornet;

namespace NeuralNetworkButGood
{
    public interface ILayer
    {
        public TensorShape LayerSize { get; }
        public abstract Tensor<float> FeedForward(Tensor<float> WorkingVector);
    }

    public interface IWeightBias
    {
        public Tensor<float> Weights { get; }
        public Tensor<float> Biases { get; }
    }
    public interface IKernel
    {
        public Tensor<float> Kernel { get; }
    }

    public interface IActivation
    {
        public Func<float, float> ActivationFunction { get; }
    }

    public class GenericLayer : ILayer, IWeightBias, IActivation
    {
        public TensorShape LayerSize => _layerSize;
        private readonly TensorShape _layerSize;


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

            _layerSize = new TensorShape(layerSize);
        }

        public Tensor<float> FeedForward(Tensor<float> WorkingVector)
        {
            WorkingVector = (WorkingVector * Weights).Sum(1) + Biases;
            WorkingVector.ForEachInplace(ActivationFunction);
            return WorkingVector;
        }
    }

    public class ConvolutionalLayer : ILayer, IKernel, IActivation
    {
        public Tensor<float> Kernel => _kernel;
        private Tensor<float> _kernel;

        public Func<float, float> ActivationFunction => _activationFunction;
        private Func<float, float> _activationFunction;

        public TensorShape LayerSize => _layerSize;
        private TensorShape _layerSize;

        public ConvolutionalLayer(int width, int length, int kernelSize, Func<float, float>? Activation = null)
        {
            this._layerSize = new TensorShape(width, length);

            _kernel = Tensor.Zeros<float>(new TensorShape(kernelSize, kernelSize));
        }

        public Tensor<float> FeedForward(Tensor<float> input)
        {
            
        }
    }

    public class SoftMaxFullConnected : ILayer, IWeightBias
    {
        public TensorShape LayerSize => _layerSize;
        private readonly TensorShape _layerSize;

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

            _layerSize = new TensorShape(layerSize);
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
        public TensorShape LayerSize => _layerSize;
        private readonly TensorShape _layerSize;

        public InputLayer(int layerSize)
        {
            _layerSize = new TensorShape(layerSize);
        }

        public Tensor<float> FeedForward(Tensor<float> WorkingVector)
        {
            return WorkingVector;
        }
    }

    public static class Activations
    {

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Tanh(float value)
        {
            return MathF.Tanh(value);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Relu(float value)
        {
            return Math.Max(value, 0);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Sigmoid(float value)
        {
            return 1f / (1f + MathF.Exp(-value));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float LeakyRelu(float value)
        {
            const float LeakValue = 0.01f;
            return value > 0 ? value : value * LeakValue;
        }
    }
}
