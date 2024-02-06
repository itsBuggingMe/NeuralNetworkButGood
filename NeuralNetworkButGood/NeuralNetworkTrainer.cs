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
using System.Numerics;

namespace NeuralNetworkButGood
{
    public class NeuralNetworkTrainer
    {
        public static NeuralNetworkFast TrainStochasticGradientDecsent(NeuralNetworkFast network, TrainingData data, int epochs, float learningRate = 0.01f)
        {
            int layers = network.Layers.Length;
            
            Tensor<float>[] WeightsGrads = new Tensor<float>[layers - 1];
            for (int l = 1; l < layers; l++)
            {
                if (network.Layers[l] is not IWeightBias)
                    throw new ArgumentException($"Layer {l} is not IWeightable");

                WeightsGrads[l - 1] = Tensor.Zeros<float>(((IWeightBias)network.Layers[l]).Weights.Shape);
            }

            IWeightBias[] WeightedRef = network.Layers.Where(c => c is IWeightBias).Cast<IWeightBias>().ToArray();
            int[] Sizes = network.Layers.Select(l => l.LayerSize.Shape[0]).ToArray();

            for (int i = 0; i < epochs; i++)
            {
                for (int layer = 0; layer < layers - 1; layer++)
                {
                    GenericLayer loc = (GenericLayer)WeightedRef[layer];

                    float currentCost = GetCostOfNetwork(network, data);
                    const float delta = 0.001f;

                    //biases
                    for (int thisL = 0; thisL < Sizes[layer+1]; thisL++)
                    {

                        float originalWeight = loc.Biases[thisL];
                        loc.Biases[thisL] += delta;

                        float newCost = GetCostOfNetwork(network, data);

                        if (newCost < currentCost)
                        {
                            loc.Biases[thisL] += learningRate;
                            currentCost = GetCostOfNetwork(network, data);
                        }
                        else if (newCost > currentCost)
                        {
                            loc.Biases[thisL] -= learningRate;
                            currentCost = GetCostOfNetwork(network, data);
                        }
                    }

                    for (int thisL = 0; thisL < Sizes[layer]; thisL++)
                    {
                        for (int nL = 0; nL < Sizes[layer + 1]; nL++)
                        {

                            float originalWeight = loc.Weights[nL, thisL];
                            loc.Weights[nL, thisL] += delta;

                            float newCost = GetCostOfNetwork(network, data);

                            if(newCost < currentCost)
                            {
                                loc.Weights[nL, thisL] += learningRate;
                                currentCost = GetCostOfNetwork(network, data);
                            }
                            else if (newCost > currentCost)
                            {
                                loc.Weights[nL, thisL] -= learningRate;
                                currentCost = GetCostOfNetwork(network, data);
                            }

                        }
                    }
                }
                /*
                for (int layer = 0; layer < layers - 1; layer++)
                {
                    WeightedRef[layer].Weights += WeightsGrads[layer];
                }*/

                //learningRate *= decay;
                //report
                //if (i % 100 == 0)
                Console.WriteLine($"Epoch: {i} NetworkCost: {GetCostOfNetwork(network, data)}");

                if(i % 10 == 0)
                {
                    SolutionSpaceVisualizer.DrawRanges(256, 256, @$"C:\Users\Jason\Downloads\resultsrbg\rbgGEN{i}.png",
                new Vector2(0, 1), new Vector2(0, 1),
                (x, y) =>
                {
                    float[] values = network.Run(new float[] { x, y });
                    return Color.FromArgb(fTB(values[0]), fTB(values[1]), fTB(values[2]));

                    byte fTB(float v)
                    {
                        return (byte)(v * 255);
                    }
                });
                }
            }

            



            return network;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float GetCostOfNetworkOne(NeuralNetworkFast network, (Tensor<float>, Tensor<float>) data)
        {
            return MeanSquaredError(
                network.Run(data.Item1),
                data.Item2
                );
        }

        //shitty code
        private static float[] Costs = new float[512];

        private static float GetCostOfNetwork(NeuralNetworkFast network, TrainingData data)
        {
            var td = data.GetAllData();

            Parallel.For(0, td.Length, i =>
            {
                float dC = MeanSquaredError(
                    network.Run(td[i].Item1),
                    td[i].Item2
                    );

                Costs[i] = dC;
            });

            return (float)(Costs.Sum() / td.Length);
        }

        /// <summary>
        /// Note: does not work for all
        /// </summary>
        private static float GetAccuracyOfNetwork(NeuralNetworkFast network, TrainingData data)
        {
            var alld = data.GetAllData();
            int correct = 0;
            for(int i = 0; i < alld.Length; i++)
            {

            }

            return (float)correct / alld.Length;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float MeanSquaredError(Tensor<float> network, Tensor<float> training)
        {
            return (network - training).ForEach((f) => { return f * f; }).Sum()[0];
        }

        public static float GetXFromLoc(in float x1, in float y1, in float x2, in float y2)
        {
            float a = (y1 * x2 - y2 * x1) / (x1 * x1 - x2 * x2 * x1);
            float b = (y2 - a * x2 * x2) / x2;

            return (-b) / (2 * a);
        }

        public static float EstimateXConst2(in float y1, in float y2)
        {
            const float range = 2;
            const float r2 = range * range;
            const float r3 = range * range * range;
            const float diffRecp = 1 / (r2 - r3);
            const float rangeRecp = 1 / range;

            float a = range * (y1 - y2) * diffRecp;
            float b = (y2 - a * r2) * rangeRecp;

            return (-b) / (2 * a);
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
