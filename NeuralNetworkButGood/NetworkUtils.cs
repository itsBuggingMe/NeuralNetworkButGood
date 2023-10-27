using System;
using System.Collections;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensornet;

namespace NeuralNetworkButGood
{
    public static class NetworkUtils
    {
        public static void Populate<T>(ref T[] arr, T value)
        {
            for(int i = 0; i < arr.Length; i++)
                arr[i] = value;
        }

        public static void Populate<T>(ref T[] arr, Func<T> value)
        {
            for (int i = 0; i < arr.Length; i++)
                arr[i] = value();
        }

        public static void Populate<T>(ref T[] arr, Func<int,T> value)
        {
            for (int i = 0; i < arr.Length; i++)
                arr[i] = value(i);
        }

        public static Tensor<float> TensorFromVector(float[] vector)
        {
            return Tensor.FromArray(vector, new int[] { vector.Length });
        }

        public static TrainingData ImageToTrainingDataBW(string[] filePaths, Func<string, float[]> GetOutputFromPath, int batchSize)
        {
            (float[], float[])[] dataRaw = new (float[], float[])[filePaths.Length];

            int thingsAdded = 0;
            foreach(string path in filePaths)
            {
                Bitmap bmp = new Bitmap(path);

                float[] input = new float[bmp.Width * bmp.Height];
                float[] output = GetOutputFromPath(path);

                int counter = 0;
                ForEachBitmap(bmp, (x,y) =>
                {
                    input[counter++] = bmp.GetPixel(x,y).R == 255 ? 0: 1;
                });
                dataRaw[thingsAdded++] = (input, output);
            }

            return new TrainingData(dataRaw, batchSize);
        }

        public static void ForEachBitmap(Bitmap bmp, Action<int, int> onForEach)
        {
            for (int x = 0; x < bmp.Width; x++)
            {
                for (int y = 0; y < bmp.Height; y++)
                {
                    onForEach(x, y);
                }
            }
        }

        public static Random GenerateRandom { get; } = new Random();

        private static Random ShuffleRandom { get; } = new Random();

        public static void ShuffleArray<T>(ref T[] collection, int? seed = null)
        {
            Random ShuffleRandom = seed.HasValue ? new Random(seed.Value) : NetworkUtils.ShuffleRandom;

            for (int i = collection.Length - 1; i >= 0; i--)
            {
                int swapIndex = ShuffleRandom.Next(i);
                var b = collection[swapIndex];

                collection[swapIndex] = collection[i];
                collection[i] = b;
            }
        }

        public static void ShuffleList<T>(ref List<T> collection, int? seed = null)
        {
            Random ShuffleRandom = seed.HasValue ? new Random(seed.Value) : NetworkUtils.ShuffleRandom;

            for (int i = collection.Count - 1; i >= 0; i--)
            {
                int swapIndex = ShuffleRandom.Next(i);
                var b = collection[swapIndex];

                collection[swapIndex] = collection[i];
                collection[i] = b;
            }
        }

        public static TimeSpan BenchmarkNetwork(INeuralNetwork network, TensorShape inputSize, in int runCount, in int rndSeed = 12345)
        {
            Random inputGenerator = new Random(rndSeed);

            Tensor InputTensor = Tensor.Zeros<float>(inputSize);
            InputTensor.ForEachInplace((f)=>
            {
                return (float)(inputGenerator.NextDouble() * 2 - 1);
            });

            //start
            DataTime a = DataTime.Now;
            for(int i = 0; i < runCount; i++)
            {
                network.FeedForward(InputTensor);
            }
            DataTime b = DataTime.Now;

            return b - a;
        }

        public static TimeSpan GenericBenchmark<I,R>(Func<I,R> GenericBenchmark, I input, in int runCount)
        {
            DataTime a = DataTime.Now;
            for(int i = 0; i < runCount; i++)
            {
                _ = GenericBenchmark(input);
            }
            DataTime b = DataTime.Now;

            return b - a;
        }
    }
}
