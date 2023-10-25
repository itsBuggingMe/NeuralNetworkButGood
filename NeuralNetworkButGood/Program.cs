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
            
            Console.WriteLine("Activation");
            string result = Console.ReadLine();

            Func<float,float> actFunc = null;
            switch(result)
            {
                case"RELU":
                    actFunc = ActivationFunctions.Relu;
                    break;
                case "SIGMOID":
                    actFunc = ActivationFunctions.Sigmoid;
                    break;
                case "TANH":
                    actFunc = ActivationFunctions.Tanh;
                    break;
                default:
                    throw new Exception();
            }

            net.AddLayer(new InputLayer(32*32));
            net.AddLayer(new GenericLayer(net.TopLayer(), 16, actFunc));
            net.AddLayer(new GenericLayer(net.TopLayer(), 16, actFunc));
            net.AddLayer(new SoftMax(net.TopLayer(), 10));

            string output = @"C:\Users\Jason\OneDrive\Desktop\AI storage Folder\NewNetWeights32x32\";
            NeuralNetworkVisualiser vis = new NeuralNetworkVisualiser(new Point(0,0), Ask("Layer"), output, Ask("Range"));

            
            Console.WriteLine("Begin Sample");
            for(int j = 0; j < 16; j++)//now
            {
                for (int i = 0; i < 16; i++)//prev
                {
                    vis.Location = new Point(j, i);
                    vis.SampleWeight(net, data);
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