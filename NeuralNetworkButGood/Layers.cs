using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkButGood
{
    internal interface ILayer
    {
        public int LayerSize { get; }
        public abstract Vector<float> FeedForward(Vector<float> WorkingVector);
    }

    internal class GenericLayer : ILayer
    {
        public int LayerSize => _layerSize;
        private readonly int _layerSize;

        private Matrix<float> Weights;
        private Vector<float> Biases;

        private Func<float, float> ActivationFunction;


        public GenericLayer(ILayer previousLayer, int layerSize, Func<float, float>? Activation = null)
        {
            this.Weights = Matrix<float>.Build.Random(layerSize, previousLayer.LayerSize);
            this.Biases = Vector<float>.Build.Random(layerSize);

            this.ActivationFunction = Activation ?? ActivationFunctions.Relu;

            _layerSize = layerSize;
        }

        public Vector<float> FeedForward(Vector<float> WorkingVector)
        {
            WorkingVector = Weights * WorkingVector + Biases;
            WorkingVector.MapInplace(ActivationFunction);
            return WorkingVector;
        }

        internal static class ActivationFunctions
        {
            public static float Atan(float value)
            {
                return MathF.Atan(value);
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

        public Vector<float> FeedForward(Vector<float> WorkingVector)
        {
            return WorkingVector;
        }
    }
}
