using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.ComponentModel;
using System.Reflection.Metadata.Ecma335;
using Tensornet;

namespace NeuralNetworkButGood
{
    public class NeuralNetworkTrainer
    {
        public static float MeanSquaredError(Tensor<float> network, Tensor<float> training)
        {
            return (network - training).ForEach((f) => { return f * f; }).Sum()[0];
        }
    }


    public class TrainingData
    {
        //input, output
        private List<(Tensor<float>, Tensor<float>)> DataVectors { get; set; } = new List<(Tensor<float>, Tensor<float>)>();

        private readonly int numPerDisplay;
        public int NumPerEpoch => numPerDisplay;

        public TrainingData(int numPerDisplay)
        {
            this.numPerDisplay = numPerDisplay;
        }

        public void AddData(Tensor<float> input, Tensor<float> output)
        {
            DataVectors.Add((input,output));
        }
        public void AddData(float[] input, float[] output)
        {
            DataVectors.Add(  (NetworkUtils.TensorFromVector(input), NetworkUtils.TensorFromVector(output))  );
        }
        public Tensor<float>[] RunNetwork(NeuralNetworkFast network, out Tensor<float>[] RealAnswers, int showing = 0)
        {
            var LRealAnswers = new Tensor<float>[numPerDisplay];
            var NetAnswers = new Tensor<float>[numPerDisplay];

            int transform = showing * numPerDisplay;

            for (int i = 0; i < numPerDisplay; i++)
            {
                int transIndex = (transform + i) % DataVectors.Count;
                NetAnswers[i] = network.Run(DataVectors[transIndex].Item1);
                LRealAnswers[i] = DataVectors[transIndex].Item2;
            }

            RealAnswers = LRealAnswers;
            return NetAnswers;
        }

        public (Tensor<float>, Tensor<float>)[] GetDataInstance()
        {
            return DataVectors.ToArray();
        }
    }
}
