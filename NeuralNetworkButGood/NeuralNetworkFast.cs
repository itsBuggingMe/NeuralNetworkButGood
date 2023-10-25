using Tensornet;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkButGood
{
    public class NeuralNetworkFast
    {
        float _cost;
        public float Cost
        {
            get
            {
                return _cost;
            }
            set
            {
                _cost = value;
            }
        }

        public List<ILayer> Layers = new List<ILayer>();
        public NeuralNetworkFast()
        {

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

        private Tensor<float> DoRun(Tensor<float> InputVector)
        {
            if (typeof(InputLayer) != Layers[0].GetType())
            {
                throw new Exception("Network does not start with an input layer");
            }

            Tensor<float> WorkingVector = Tensor.FromEnumerable(InputVector, InputVector.Shape.ToArray());
            foreach (ILayer layer in Layers)
            {
                WorkingVector = layer.FeedForward(WorkingVector);
            }

            return WorkingVector;
        }

        public void AddLayer(ILayer layer)
        {
            Layers.Add(layer);
        }

        public ILayer TopLayer()
        {
            return Layers[Layers.Count - 1];
        }
    }
}
