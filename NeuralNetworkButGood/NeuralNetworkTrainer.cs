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
using System.Reflection.Metadata.Ecma335;

namespace NeuralNetworkButGood
{
    internal class NeuralNetworkTrainer
    {
        public static NeuralNetworkFast TrainNetworkGeneticAlgorithm(int epochs, TrainingData data, int numNetworks, Func<NeuralNetworkFast> getInitalNet,
            float mutationRange = 0.08f, float mutationRate = 0.04f)
        {
            NeuralNetworkFast[] nets = new NeuralNetworkFast[numNetworks];
            for(int i = 0; i < numNetworks; i++)
            {
                nets[i] = getInitalNet();
            }

            for(int e = 0; e < epochs; e++)
            {
                Console.WriteLine("Starting Epoch " + e);

                for (int i = 0; i < numNetworks; i++)
                {
                    nets[i].Cost = 0;
                    var netAnswers = data.RunNetwork(nets[i], out var realAnswers);
                    for(int j = 0; j < netAnswers.Length; j++)
                    {
                        nets[i].Cost += MeanSquaredError(realAnswers[j], netAnswers[j]);
                    }
                }

                Array.Sort(nets, (a,b) => {
                    if(a.Cost > b.Cost)
                    {
                        return 1;
                    }
                    else if (b.Cost > a.Cost)
                    {
                        return -1;
                    }
                    else
                    {
                        return 0;
                    }
                });

                Console.WriteLine($"Best Cost: {nets[0].Cost}");
                Console.WriteLine($"Worst Cost: {nets[numNetworks-1].Cost}");

                for (int i = 1; i < numNetworks; i++)
                {
                    var mutateFrom = nets[0];
                    var mutatedNetwork = new NeuralNetworkFast();

                    for(int j = 0; j < mutateFrom.Layers.Count; j++)
                    {
                        var Layer = mutateFrom.Layers[j].CopyOf();
                        if(Layer.GetType() == typeof(GenericLayer))
                            (Layer as GenericLayer).Mutate(mutationRange, mutationRate);

                        mutatedNetwork.AddLayer(Layer);
                    }

                    nets[i] = mutatedNetwork;
                }
            }

            return nets[0];
        }

        private static float MeanSquaredError(Vector<float> real, Vector<float> cap)
        {
            /*
            var difference = real - cap;
            difference.PointwiseAbs();
            float sum =  difference.Sum();*/
            return Math.Abs(real[0] - cap[0]);
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

            for(int i = 0; i < DataVectors.Count; i++)
            {
                NetAnswers[i] = network.Run(DataVectors[i].Item1);
                LRealAnswers[i] = DataVectors[i].Item2;
            }

            RealAnswers = LRealAnswers;
            return NetAnswers;
        }
    }
}
