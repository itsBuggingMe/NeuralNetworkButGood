using System.IO;
using System.Drawing;
using Tensornet;
using System;
using System.Windows;
using MNIST.IO;
using System.Security.Cryptography;
using Microsoft.VisualBasic;

namespace NeuralNetworkButGood
{
    internal class Program
    {
        const string output = @"C:\Users\Jason\OneDrive\Desktop\AI storage Folder\NewNetWeights32x32\";

        private static readonly string[,] dataFilePaths =
        {
           { @"G:\Shared drives\TRAINNING DATA\Mnist Dataset\train-images.idx3-ubyte", @"G:\Shared drives\TRAINNING DATA\Mnist Dataset\train-labels.idx1-ubyte" },
           { @"G:\Shared drives\TRAINNING DATA\Mnist Dataset\t10k-images.idx3-ubyte", @"G:\Shared drives\TRAINNING DATA\Mnist Dataset\t10k-labels.idx1-ubyte" }
        };

        static void Main()
        {
            //TODO:
            //Convolutional
            //Gradient, Schotacisc GD, Adagrad? grav?
            //GAN image gen?

            NeuralNetworkFast neuralNetwork = new NeuralNetworkFast(3);
            neuralNetwork.SetLayer(0, new InputLayer(2));
            neuralNetwork.SetLayer(1, new GenericLayer(2, 8));
            neuralNetwork.SetLayer(2, new SoftMaxFullConnected(8, 3));

            NeuralNetworkVisualiser.
        }



        /*
        public static int Ask(string msg)
        {
            Console.WriteLine(msg);
            return int.Parse(Console.ReadLine());
        }

        public static void TrainingTest(TrainingData data)
        {
            NeuralNetworkFast net = new NeuralNetworkFast(4);

            net.SetLayer(0, new InputLayer(32 * 32));
            net.SetLayer(1, new GenericLayer(net.Layers[0].LayerSize, 16, Activations.Sigmoid));
            net.SetLayer(2, new GenericLayer(net.Layers[1].LayerSize, 16, Activations.Sigmoid));
            net.SetLayer(3, new SoftMaxFullConnected(net.Layers[2].LayerSize, 10));

            NeuralNetworkTrainer.TrainStochasticGradientDecsent(net, data, 1000);
        }

        public static TrainingData GetImages()
        {
            string[] paths = Directory.GetFiles(@"G:\Shared drives\TRAINNING DATA\compressed 32x32");

            TrainingData data = NetworkUtils.ImageToTrainingDataBW(paths,
                (path) =>
                {
                    float[] ans = new float[10];
                    ans[path[49] - '0'] = 1;
                    return ans;
                },
                paths.Length
                );

            return data;
        }

        public static (TrainingDataLite, TrainingDataLite) GetMnistDataset()
        {
            TestCase[] dataTrain = FileReaderMNIST.LoadImagesAndLables(dataFilePaths[0, 0], dataFilePaths[0, 1]).ToArray();
            TestCase[] dataTest = FileReaderMNIST.LoadImagesAndLables(dataFilePaths[1, 0], dataFilePaths[1, 1]).ToArray();

            var inputImageShape = new TensorShape(28, 28);

            var tensorsTrain = LoadDataAndCreateTensors(dataTrain, inputImageShape);
            var tensorsTest = LoadDataAndCreateTensors(dataTest, inputImageShape);

            return (new TrainingDataLite(tensorsTrain), new TrainingDataLite(tensorsTest));
        }

        private static (Tensor<float>, Tensor<float>)[] LoadDataAndCreateTensors(TestCase[] data, TensorShape inputImageShape)
        {
            var tensors = new (Tensor<float>, Tensor<float>)[data.Length];

            for (int i = 0; i < data.Length; i++)
            {
                float[] ansArr = new float[10];
                ansArr[data[i].Label] = 1;

                float[] imageArr = new float[28 * 28];
                for (int p = 0; p < imageArr.Length; p++)
                {
                    imageArr[p] = data[i].Image[p / 28, p % 28] / 255f;
                }

                tensors[i] = (Tensor.FromArray(imageArr, inputImageShape), NetworkUtils.TensorFromVector(ansArr));
            }

            return tensors;
        }*/
    }
}