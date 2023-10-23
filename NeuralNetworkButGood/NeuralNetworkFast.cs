using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkButGood
{
    internal class NeuralNetworkFast
    {
        public float Cost = 0;
        private List<ILayer> Layers = new List<ILayer>();
        public NeuralNetworkFast()
        {

        }

        public Vector<float> Run(Vector<float> InputVector)
        {
            if(typeof(InputLayer) != Layers[0].GetType())
            {
                throw new Exception("Network does not start with an input layer");
            }

            Vector<float> WorkingVector = Vector<float>.Build.SameAs(InputVector);
            InputVector.CopyTo(WorkingVector);

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
