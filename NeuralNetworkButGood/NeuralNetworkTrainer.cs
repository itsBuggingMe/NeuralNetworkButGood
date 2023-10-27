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
            Tensor<float>[] Weights = new Tensor<float>[network.Layers.Length - 1];
            Tensor<float>[] Gradients = new Tensor<float>[network.Layers.Length - 1];

            for (int l = 1; l < network.Layers.Length; l++)
            {
                if (network.Layers[l] is not IWeightable)
                    throw new ArgumentException($"Layer {l} is not IWeightable");

                Weights[l - 1] = ((IWeightable)network.Layers[l]).Weights;
                Gradients[l - 1] = Tensor.Zeros<float>(Weights[l - 1].Shape);
            }

            for (int i = 0; i < epochs; i++)
            {
                (Tensor<float>, Tensor<float>) trainingData = data.GetDataInstance();

                Tensor<float> NetworkOutput = network.Run(trainingData.Item1);

                for (int LayerIndex = Weights.Length; LayerIndex > 0; LayerIndex--)
                {// do back prop
                    Tensor<float> diffVector = trainingData.Item2 - NetworkOutput;

                    for(int NeuronDifferenceIndex = 0; NeuronDifferenceIndex < diffVector.)
                    {

                    }
                }
                //apply gradients



                //report
                if(i % 10 == 0)
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
            DoInitalise(DataVectors, batchSize, RemoveExtraData);
        }

        public TrainingData((float[], float[])[] DataVectors, int batchSize, bool RemoveExtraData = true)
        {
            (Tensor<float>, Tensor<float>)[] DataTensors = new (Tensor<float>, Tensor<float>)[batchSize];

            for (int i = 0; i < DataVectors.Length; i++)
                DataTensors[i] = (NetworkUtils.TensorFromVector(DataVectors[i].Item1), NetworkUtils.TensorFromVector(DataVectors[i].Item2));

            DoInitalise(DataTensors, batchSize,RemoveExtraData);
        }

        private void DoInitalise((Tensor<float>, Tensor<float>)[] DataVectors, int batchSize, bool RemoveExtraData)
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
                for(int j = 0; j < batchSize; j++)
                {
                    DataVectorsBatched[i][j] = DataVectors[i * batchSize + j];
                }
            }
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
}
