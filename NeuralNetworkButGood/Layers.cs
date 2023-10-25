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
        public ILayer CopyOf();
    }

    public class GenericLayer : ILayer
    {
        public int LayerSize => _layerSize;
        private readonly int _layerSize;

        private readonly int _prevLayerSize;

        public Tensor<float> Weights;
        public Tensor<float> Biases;

        private Func<float, float> ActivationFunction;


        /// <summary>
        /// Default Activation is Relu
        /// </summary>
        /// <param name="previousLayer"></param>
        /// <param name="layerSize"></param>
        /// <param name="Activation"></param>
        public GenericLayer(ILayer previousLayer, int layerSize, Func<float, float>? Activation = null)
        {
            this.Weights = Tensor.Zeros<float>(new TensorShape(layerSize, previousLayer.LayerSize));
            this.Biases = Tensor.Zeros<float>(new TensorShape(layerSize));

            Weights.ForEachInplace((f) => { return (float)NetworkUtils.GenerateRandom.NextDouble() * 2 - 1; });

            Biases.ForEachInplace((f) => { return (float)NetworkUtils.GenerateRandom.NextDouble() * 2 - 1; });
            


            this.ActivationFunction = Activation ?? ActivationFunctions.Relu;

            _layerSize = layerSize;
            _prevLayerSize = previousLayer.LayerSize;
        }

        private GenericLayer(int prevLayerSize, int layerSize, Func<float, float>? Activation = null)
        {
            Weights.ForEachInplace((f) => { return (float)NetworkUtils.GenerateRandom.NextDouble() * 2 - 1; });
            Biases.ForEachInplace((f) => { return (float)NetworkUtils.GenerateRandom.NextDouble() * 2 - 1; });

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
    }

    public class SoftMax : ILayer
    {
        public int LayerSize => _layerSize;
        private readonly int _layerSize;

        private readonly int _prevLayerSize;

        public Tensor<float> Weights;
        public Tensor<float> Biases;

        public SoftMax(ILayer previousLayer, int layerSize)
        {
            this.Weights = Tensor.Random.Uniform<float>(new TensorShape(layerSize, previousLayer.LayerSize));
            this.Biases = Tensor.Random.Uniform<float>(new TensorShape(layerSize));

            _layerSize = layerSize;
            _prevLayerSize = previousLayer.LayerSize;
        }

        private SoftMax(int prevLayerSize, int layerSize)
        {
            this.Weights = Tensor.Random.Uniform<float>(new TensorShape(prevLayerSize, layerSize));
            this.Biases = Tensor.Random.Uniform<float>(new TensorShape(layerSize));

            _prevLayerSize = prevLayerSize;
            _layerSize = layerSize;
        }

        public Tensor<float> FeedForward(Tensor<float> WorkingVector)
        {
            WorkingVector = ((WorkingVector * Weights).Sum(1) + Biases).ForEach(ActivationFunctions.Sigmoid);

            float sum = WorkingVector.Sum()[0];
            float inverseSum = 1f / sum;
            return WorkingVector * inverseSum;
        }

        public ILayer CopyOf()
        {
            var copy = new SoftMax(_prevLayerSize, LayerSize);
            Weights.CopyTo(copy.Weights);
            Biases.CopyTo(copy.Biases);
            return copy;
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

        public ILayer CopyOf()
        {
            return new InputLayer(_layerSize);
        }
    }

    public static class ActivationFunctions
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
