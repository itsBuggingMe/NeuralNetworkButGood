using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.ComponentModel;
using System.Reflection.Metadata.Ecma335;
using Tensornet;
using System.Runtime.CompilerServices;

namespace NeuralNetworkButGood
{
    public class NeuralNetworkTrainer
    {
        public static NeuralNetworkFast TrainStochasticGradientDecsent(NeuralNetworkFast network, TrainingData data, int epochs, float learningRate = 0.05f)
        {
            int layers = network.Layers.Length;

            Tensor<float>[] Weights = new Tensor<float>[layers - 1];

            for (int l = 1; l < layers; l++)
            {
                if (network.Layers[l] is not IWeightBias)
                    throw new ArgumentException($"Layer {l} is not IWeightable");

                Weights[l - 1] = ((IWeightBias)network.Layers[l]).Weights;
            }

            int[] Sizes = Weights.Select(w => w.Shape[0]).ToArray();

            for (int i = 0; i < epochs; i++)
            {
                (Tensor<float>, Tensor<float>) trainingData = data.GetDataInstance();

                Tensor<float> NetworkOutput = network.RunCapture(trainingData.Item1, out Tensor<float>[] neuronData);

                Tensor<float> thisNeuronGradients = trainingData.Item2 - neuronData[layers - 1];

                Tensor<float> error = NetworkOutput - trainingData.Item2;
                Tensor<float>[] gradients = network.Backpropagate(error, neuronData);

                for (int l = 0; l < Weights.Length; l++)
                {
                    Weights[l] -= learningRate * gradients[l];
                    Weights[l].ForEachInplace(f => Random.Shared.NextSingle() * 2 - 1);
                }


                //report
                if (i % 20 == 0)
                    Console.WriteLine($"Epoch: {i} NetworkCost: {GetCostOfNetwork(network, data)}");
            }

            return network;
        }


        private static float GetCostOfNetwork(NeuralNetworkFast network, TrainingData data)
        {
            var td = data.GetAllData();
            float cost = 0;
            foreach (var tuple in td)
            {
                float dC = MeanSquaredError(
                    network.Run(tuple.Item1),
                    tuple.Item2
                    );
                cost += dC;
            }

            return (float)(cost / td.Length);
        }

        public static float MeanSquaredError(Tensor<float> network, Tensor<float> training)
        {
            return (network - training).ForEach((f) => { return f * f; }).Sum()[0];
        }
    }


    public class TrainingData
    {
        private (Tensor<float>, Tensor<float>)[] DataVectors;
        private (Tensor<float>, Tensor<float>)[][] DataVectorsBatched;

        public int BatchSize => batchSize;
        private readonly int batchSize;

        public int TotalDataCount => DataVectors.Length;

        private int BatchIndex = 0;
        public int BatchCount => DataVectorsBatched.Length;

        private int instanceDataGetIndex = 0;

        public TrainingData((Tensor<float>, Tensor<float>)[] DataVectors, int batchSize, bool RemoveExtraData = true)
        {
            if (!RemoveExtraData && (DataVectors.Length % batchSize != 0))
                throw new ArgumentException($"The length of DataVectors ({DataVectors.Length}) is not divisible by batch size ({batchSize})");

            NetworkUtils.ShuffleArray(ref DataVectors);

            this.DataVectors = DataVectors;

            int batchCount = DataVectors.Length / batchSize;

            DataVectorsBatched = new (Tensor<float>, Tensor<float>)[batchCount][];

            for (int i = 0; i < batchCount; i++)
            {
                DataVectorsBatched[i] = new (Tensor<float>, Tensor<float>)[batchSize];
                for (int j = 0; j < batchSize; j++)
                {
                    DataVectorsBatched[i][j] = DataVectors[i * batchSize + j];
                }
            }
        }

        public TrainingData((float[], float[])[] DataVectors, int batchSize, bool RemoveExtraData = true) : this(Transform(DataVectors), batchSize, RemoveExtraData)
        {
            
        }

        static (Tensor<float>, Tensor<float>)[] Transform((float[], float[])[] DataVectors)
        {
            (Tensor<float>, Tensor<float>)[] DataTensors = new (Tensor<float>, Tensor<float>)[DataVectors.Length];

            for (int i = 0; i < DataVectors.Length; i++)
                DataTensors[i] = (NetworkUtils.TensorFromVector(DataVectors[i].Item1), NetworkUtils.TensorFromVector(DataVectors[i].Item2));

            return DataTensors;
        }

        public (Tensor<float>, Tensor<float>) GetDataInstance()
        {
            return DataVectors[instanceDataGetIndex++ % DataVectors.Length];
        }

        public (Tensor<float>, Tensor<float>)[] GetDataBatch()
        {
            BatchIndex %= BatchCount;
            return DataVectorsBatched[BatchIndex++];
        }

        public (Tensor<float>, Tensor<float>)[] GetDataBatchIndex(int index)
        {
            if(index < 0 || index >= BatchCount)
                throw new ArgumentOutOfRangeException("index");

            return DataVectorsBatched[index];
        }

        public (Tensor<float>, Tensor<float>)[] GetAllData()
        {
            return DataVectors;
        }
    }

    public class TrainingDataLite
    {
        private (Tensor<float>, Tensor<float>)[] DataVectors;

        public int DataCount => DataVectors.Length;

        public TrainingDataLite((Tensor<float>, Tensor<float>)[] DataVectors)
        {
            this.DataVectors = DataVectors;

            NetworkUtils.ShuffleArray(ref this.DataVectors);
        }

        public void Shuffle()
        {
            NetworkUtils.ShuffleArray(ref DataVectors);
        }

        public (Tensor<float>, Tensor<float>) GetSample(int? index = null)
        {
            return DataVectors[index.HasValue ? index.Value : Random.Shared.Next(DataCount)];
        }

        public (Tensor<float>, Tensor<float>)[] GetBatch(int size)
        {
            int startIndex = Random.Shared.Next(DataCount - size);
            return DataVectors[startIndex..(startIndex+size)];
        }
    }

}
