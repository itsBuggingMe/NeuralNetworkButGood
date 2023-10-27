﻿using System.IO;
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
            //RELU & Leaky RELU
            //Gradient, Schotacisc GD, Adagrad? grav?
            //GAN image gen?

            string[] paths = Directory.GetFiles(@"G:\Shared drives\TRAINNING DATA\compressed 32x32");
            /*
            TrainingData data = NetworkUtils.ImageToTrainingDataBW(paths,
                (path) => {
                    float[] ans = new float[10];
                    ans[path[49] - '0'] = 1;
                    return ans;
                }, 
                paths.Length
                );*/

            Console.WriteLine("Images Loaded");

            //TrainingTest(data);
            //return;

            NeuralNetworkFast net = new NeuralNetworkFast(4);

            net.SetLayer(0, new InputLayer(32 * 32));
            net.SetLayer(1, new GenericLayer(net.Layers[0].LayerSize, 16, Activations.Relu));
            net.SetLayer(2, new GenericLayer(net.Layers[1].LayerSize, 16, Activations.Relu));
            net.SetLayer(3, new SoftMaxFullConnected(net.Layers[2].LayerSize, 10));

            TimeSpan[] time = NetworkUtils.MultiBenchmarkNetwork(net, new TensorShape(32 * 32),10, 10000);

            double total = 0;

            foreach(var t in time)
            {
                Console.WriteLine(t.TotalMilliseconds);
                total += t.TotalMilliseconds;
            }

            Console.WriteLine("+++++++++++++++++++++++++++++++++++");

            NeuralNetworkFast net2 = new NeuralNetworkFast(4);

            net2.SetLayer(0, new InputLayer(64));
            net2.SetLayer(1, new GenericLayer(net2.Layers[0].LayerSize, 64, Activations.Relu));
            net2.SetLayer(2, new GenericLayer(net2.Layers[1].LayerSize, 64, Activations.Relu));
            net2.SetLayer(2, new GenericLayer(net2.Layers[1].LayerSize, 64, Activations.Relu));
            net2.SetLayer(3, new SoftMaxFullConnected(net2.Layers[2].LayerSize, 10));

            TimeSpan[] time2 = NetworkUtils.MultiBenchmarkNetwork(net2, new TensorShape(64), 10, 10000);


            foreach (var t in time2)
            {
                Console.WriteLine(t.TotalMilliseconds);
                total += t.TotalMilliseconds;
            }

            Console.WriteLine("Total: " + total);

            Console.ReadLine();
            string output = @"C:\Users\Jason\OneDrive\Desktop\AI storage Folder\NewNetWeights32x32\";
            NeuralNetworkVisualiser vis = new NeuralNetworkVisualiser(new Point(3,8), 3, output, 16);

            
            Console.WriteLine("Begin Sample");

            for(int i = 0; i < 8; i++)
            {
                vis.Location = new Point(i, vis.Location.Y);
                vis.SampleWeight(net, null);
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