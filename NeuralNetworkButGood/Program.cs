using System.IO;
using System.Drawing;
using MathNet.Numerics.LinearAlgebra;
using Tensornet;

namespace NeuralNetworkButGood
{
    internal class Program
    {
        static void Main()
        {
            TrainingData trainingData = new TrainingData(50);

            Tensor<float> test = Tensor.FromArray(new float[] { 1,2,3,4,5,6 }, new TensorShape(6));
            Tensor<float> test2 = Tensor.FromArray(new float[] { 1,2,3,4,5,6,8,9,7,10,11,12 }, new TensorShape(2,6));

            var mult = test * test2;

            //string[] filePaths = Directory.GetFiles(@"G:\Shared drives\TRAINNING DATA\compressed 32x32");
            
            NeuralNetworkFast net = new NeuralNetworkFast();
            net.AddLayer(new InputLayer(2));
            net.AddLayer(new GenericLayer(net.TopLayer(), 64));
            net.AddLayer(new GenericLayer(net.TopLayer(), 64));
            net.AddLayer(new GenericLayer(net.TopLayer(), 64));
            net.AddLayer(new GenericLayer(net.TopLayer(), 64));
            const int Tests = (int)(10e4);
            var inputVector = NetworkUtils.TensorFromVector(new float[] {0.05f,0.95f });

            DateTime a = DateTime.Now;
            for(int i = 0; i < Tests; i++)
            {
                net.Run(inputVector);
            }
            TimeSpan b = DateTime.Now - a; 
            Console.WriteLine($"Total Time: {b.TotalMilliseconds}");
        }

        public static void GenerateImage(Func<(float, float), float> Network, string output, int imageSize, float scaleFactor)
        {
            Bitmap bitmap = new Bitmap(imageSize, imageSize);
            float halfImageSize = imageSize * 0.5f;

            for (int x = 0; x < imageSize; x++)
            {
                for (int y = 0; y < imageSize; y++)
                {
                    float netOutput = Network(
                        ((x - halfImageSize) * scaleFactor, (y - halfImageSize) * scaleFactor)
                        );

                    byte val = (byte)(netOutput * 255f);

                    bitmap.SetPixel(x, y, Color.FromArgb(val, val, val));
                }
            }

            bitmap.Save(output);
        }

        public static float GetRandomNum()
        {
            return (float)(Random.Shared.NextDouble() - 0.5f) * 2;
        }

        public static void ForEachPixelInBitmap(Bitmap bitmap, Action<int, int> ForEach)
        {
            for (int x = 0; x < bitmap.Width; x++)
            {
                for (int y = 0; y < bitmap.Height; y++)
                {
                    ForEach(x,y);
                }
            }
        }
    }
}