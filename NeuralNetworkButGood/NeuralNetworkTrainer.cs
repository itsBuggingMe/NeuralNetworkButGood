using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics;
using MathNet;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using System.Drawing;
using System.ComponentModel;

namespace NeuralNetworkButGood
{
    internal class NeuralNetworkTrainer
    {
        public static NeuralNetworkFast TrainNetworkGeneticAlgorithm(int epochs, TrainingData data, int numNetworks, Func<NeuralNetworkFast> getInitalNet,
            float mutationRange = 0.08f, float mutationRate = 0.02f)
        {
            NeuralNetworkFast[] nets = new NeuralNetworkFast[numNetworks];
            for(int i = 0; i < numNetworks; i++)
            {
                nets[i] = getInitalNet();
            }

            for(int e = 0; e < epochs; e++)
            {
                for(int i = 0; i < numNetworks; i++)
                {
                    var netAnswers = data.RunNetwork(nets[i], out var realAnswers);
                    for(int j = 0; j < netAnswers.Length; j++)
                    {
                        nets[i].Cost += MeanSquaredError(realAnswers[j], netAnswers[j]);
                    }
                }

                Array.Sort(nets, (a,b) => { return (int)(a.Cost - b.Cost); });

                for(int i = 1; i < numNetworks; i++)
                {
                    nets[i] = 
                }
            }
        }

        private static float MeanSquaredError(Vector<float> real, Vector<float> cap)
        {
            var difference = real - cap;
            difference.PointwiseMultiply(difference);
            return difference.Sum();
        }
    }


    internal class TrainingData
    {
        //input, output
        public List<(Vector<float>, Vector<float>)> DataVectors { get; set; } = new List<(Vector<float>, Vector<float>)>();

        public TrainingData()
        {

        }

        public void AddData(Vector<float> input, Vector<float> output)
        {
            DataVectors.Add((input,output));
        }

        public Vector<float>[] RunNetwork(NeuralNetworkFast network, out Vector<float>[] RealAnswers)
        {
            var LRealAnswers = new Vector<float>[DataVectors.Count];
            var NetAnswers = new Vector<float>[DataVectors.Count];

            Parallel.For(0, DataVectors.Count, (i) =>
            {
                NetAnswers[i] = network.Run(DataVectors[i].Item1);
                LRealAnswers[i] = DataVectors[i].Item2;
            });
            RealAnswers = LRealAnswers;
            return NetAnswers;
        }
    }
}
