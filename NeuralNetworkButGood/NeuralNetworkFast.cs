using Tensornet;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Reflection.Emit;
using System.Drawing.Printing;

namespace NeuralNetworkButGood
{
    public interface INeuralNetwork
    {
        public Tensor<float> Run(Tensor<float> input);
        public float[] Run(float[] input);
    }

    public interface ILayeredNetwork : INeuralNetwork
    {
        public ILayer[] Layers { get; }
    }

    public class NeuralNetworkFast : ILayeredNetwork
    {
        private bool _init = false;

        public ILayer[] Layers => _layers;
        private ILayer[] _layers;

        public NeuralNetworkFast(int LayerCount)
        {
            _layers = new ILayer[LayerCount];
        }

        public Tensor<float> Run(Tensor<float> InputVector)
        {
            return DoRun(InputVector);
        }

        public float[] Run(float[] InputVector)
        {
            var input = Tensor.FromArray(InputVector, new int[] { InputVector.Length });
            return DoRun(input).ToArray();
        }

        public Tensor<float> RunCapture(Tensor<float> InputVector, out Tensor<float>[] Neurons)
        {
            Tensor<float> WorkingVector = Tensor.FromEnumerable(InputVector, InputVector.Shape.ToArray());

            Neurons = new Tensor<float>[_layers.Length];

            for(int i = 0; i < _layers.Length; i++)
            {
                WorkingVector = _layers[i].FeedForward(WorkingVector);

                Neurons[i] = Tensor.Zeros<float>(WorkingVector.Shape);
                WorkingVector.CopyTo(Neurons[i]);
            }

            return WorkingVector;
        }

        private Tensor<float> DoRun(Tensor<float> InputVector)
        {
            Tensor<float> WorkingVector = Tensor.FromEnumerable(InputVector, InputVector.Shape.ToArray());
            foreach (ILayer layer in Layers)
            {
                WorkingVector = layer.FeedForward(WorkingVector);
            }

            return WorkingVector;
        }

        public void SetLayer(int index, ILayer layer)
        {
            Layers[index] = layer;
        }
    }
}
