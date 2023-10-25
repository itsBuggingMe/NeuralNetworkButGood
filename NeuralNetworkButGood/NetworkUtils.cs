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

        public static TrainingData ImageToTrainingDataBW(string[] filePaths, Func<string, float[]> GetOutputFromPath, int numPerDisplay)
        {
            TrainingData allData = new TrainingData(numPerDisplay);

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
                allData.AddData(input, output);
            }

            return allData;
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
    }
}
