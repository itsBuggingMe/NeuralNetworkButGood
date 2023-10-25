using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensornet;

namespace NeuralNetworkButGood
{
    internal static class NetworkUtils
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
    }
}
