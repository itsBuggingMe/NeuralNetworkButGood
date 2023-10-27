using System.IO;
using System.Drawing;
using Tensornet;
using System;
using System.Windows;

namespace NeuralNetworkButGood
{
    internal class Program
    {
        static void Main()
        {
            //TODO:
            //Convolutional
            //Unit testing as Util
            //RELU & Leaky RELU
            //Gradient, Schotacisc GD, Adagrad? grav?
            //GAN image gen?
            //mage

            string[] paths = Directory.GetFiles(@"G:\Shared drives\TRAINNING DATA\compressed 32x32");
            
            TrainingData data = NetworkUtils.ImageToTrainingDataBW(paths,
                (path) => {
                    float[] ans = new float[10];
                    ans[path[49] - '0'] = 1;
                    return ans;
                }, 
                paths.Length
                );

            Console.WriteLine("Images Loaded");

            TrainingTest(data);
            return;

            NeuralNetworkFast net = new NeuralNetworkFast(4);

            net.SetLayer(0, new InputLayer(32 * 32));
            net.SetLayer(1, new GenericLayer(net.Layers[0].LayerSize, 16, Activations.Sigmoid));
            net.SetLayer(2, new GenericLayer(net.Layers[1].LayerSize, 16, Activations.Sigmoid));
            net.SetLayer(3, new SoftMaxFullConnected(net.Layers[2].LayerSize, 10));


            string output = @"C:\Users\Jason\OneDrive\Desktop\AI storage Folder\NewNetWeights32x32\";
            NeuralNetworkVisualiser vis = new NeuralNetworkVisualiser(new Point(3,8), 3, output, 16);

            
            Console.WriteLine("Begin Sample");

            for(int i = 0; i < 8; i++)
            {
                vis.Location = new Point(i, vis.Location.Y);
                vis.SampleWeight(net, data);
            }

            Console.WriteLine("Done");
            Console.ReadLine();
        }

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
    }
}