using System.Drawing.Imaging;
using System.Drawing;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetworkButGood
{
    internal class Program
    {
        static void Main()
        {
            TrainingData trainingData = new TrainingData();

            for(int i = 0; i < 1028; i++)
            {
                float x = GetRandomNum();
                float y = GetRandomNum();

                trainingData.AddData(
                    Vector<float>.Build.DenseOfArray(new float[] { x,y }),
                    Vector<float>.Build.DenseOfArray(new float[] {
                        //System.Numerics.Vector2.Distance(new System.Numerics.Vector2(x,y), System.Numerics.Vector2.Zero) < 6f ? 1 : 0 })
                        x < 0 ? 1 : 0 })
                    );
            }

            Console.WriteLine("DataGenerated");

            var trainedNet = NeuralNetworkTrainer.TrainNetworkGeneticAlgorithm(100, trainingData,128, () =>
            {
                var net = new NeuralNetworkFast();
                net.AddLayer(new InputLayer(2));

                net.AddLayer(new GenericLayer(net.TopLayer(), 4, GenericLayer.ActivationFunctions.Sigmoid));

                net.AddLayer(new GenericLayer(net.TopLayer(), 1, GenericLayer.ActivationFunctions.Sigmoid));
                return net;
            });

            GenerateImage((v) =>
            {
                return trainedNet.Run(Vector<float>.Build.DenseOfArray(new float[] { v.Item1, v.Item2 }))[0];
            }, @"G:\My Drive\APCSA\C#\netOutput.png", 256, 1/16f);


            while (true)
            {
                int a = Convert.ToInt32(Console.ReadLine());
                int b = Convert.ToInt32(Console.ReadLine());
                float[] output = trainedNet.Run(new float[] { a, b });
                Console.WriteLine(output[0]);
            }
            /*
            var net = new NeuralNetwork();
            net.AddLayer(2);
            net.AddLayer(16);
            net.AddLayer(16);
            net.AddLayer(16);
            net.AddLayer(1);

            List<TrainingDataInstance> trainingData = new List<TrainingDataInstance>();

            for(int i = 0; i < 400; i++)
            {
                Vector2 point = new Vector2(getRandom(), getRandom());
                int value = Vector2.Distance(point, Vector2.Zero) > 6 ? 0 : 1;
                trainingData.Add(new TrainingDataInstance(new float[] {point.X, point.Y }, new float[] { value }));
            }

            var trainer = new NeuralNetworkTrainer(trainingData.ToArray());
            var trainedNet = trainer.TrainNetworkGenericAlgorithm(net, 40, 25);

            const int ImageDim = 64;

            Bitmap bitmap = new Bitmap(ImageDim, ImageDim);
            for(int x = 0; x < ImageDim; x++)
            {
                for (int y = 0; y < ImageDim; y++)
                {
                    float[] netOutput = trainedNet.Run(new float[] { XimageToNet(x, ImageDim), XimageToNet(y, ImageDim) });
                    Console.WriteLine($"{x},{y}:{netOutput[0]}");
                    byte val = (byte)(Math.Atan(netOutput[0]) * 255f);
                    bitmap.SetPixel(x,y, Color.FromArgb(val, val, val));
                }
            }

            bitmap.Save(@"C:\Users\Jason\Downloads\netOutput.png");
            Console.ReadLine();*/
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
    }
}