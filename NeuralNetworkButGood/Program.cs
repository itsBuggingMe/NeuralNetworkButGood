using System.IO;
using System.Drawing;
using Tensornet;
using System;
using System.Windows;

namespace NeuralNetworkButGood
{
    internal class Program
    {
        [STAThread]
        static void Main()
        {
            //TODO:
            //Write IBiasLayer
            //Write IWeightLayer
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

            NeuralNetworkFast net = new NeuralNetworkFast();


            net.AddLayer(new InputLayer(32*32));
            net.AddLayer(new GenericLayer(net.TopLayer(), 16, ActivationFunctions.Sigmoid));
            net.AddLayer(new GenericLayer(net.TopLayer(), 16, ActivationFunctions.Sigmoid));
            net.AddLayer(new SoftMax(net.TopLayer(), 10));

            string output = @"C:\Users\Jason\OneDrive\Desktop\AI storage Folder\NewNetWeights32x32\";
            NeuralNetworkVisualiser vis = new NeuralNetworkVisualiser(new Point(8,8), 3, output, 16);

            
            Console.WriteLine("Begin Sample");

            SoftMax layer = (SoftMax)net.Layers[3];


            for (int j = 0; j < 24; j++)//now
            {
                layer.Weights[8, 9] = (j - 12f) / 1.5f;

                vis.SampleWeight(net, data, " " + j);

                for (int i = 0; i < 16; i++)//prev
                {
                }
            }


            Console.WriteLine("Done");
            Console.ReadLine();
        }

        public static int Ask(string msg)
        {
            Console.WriteLine(msg);
            return int.Parse(Console.ReadLine());
        }
    }
}