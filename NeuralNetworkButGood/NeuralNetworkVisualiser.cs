using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensornet;

namespace NeuralNetworkButGood
{
    [System.Runtime.Versioning.SupportedOSPlatform("windows")]
    public class NeuralNetworkVisualiser
    {
        private int layer;
        public Point Location { get; set; }
        private string outputPath;
        private GraphRenderer graphRenderer;
        private float range;
        public NeuralNetworkVisualiser(Point Location, int layer, string outputPath, float range = 2)
        {
            this.range = range;
            this.layer = layer;
            this.Location = Location;
            this.outputPath = outputPath;
            this.graphRenderer = new GraphRenderer(new Point(256, 256), Color.Gray, Color.DarkCyan, Color.LightCyan);
        }

        public void SampleWeight(NeuralNetworkFast neuralNetwork, TrainingData data,string append, int pointsToSample = 128)
        {
            ILayer activeLayer = neuralNetwork.Layers[layer];

            (Tensor<float>, Tensor<float>)[] TrainingDataRaw = data.GetDataInstance();

            graphRenderer.GenerateImageFromFunction(pointsToSample, GeneratePath(append),
            (newWeightValue) =>
            {
                SoftMax layer = (SoftMax)neuralNetwork.Layers[this.layer];

                layer.Weights[Location.X, Location.Y] = newWeightValue;

                float cost = 0;
                
                foreach (var tuple in TrainingDataRaw)
                {
                    float dC = NeuralNetworkTrainer.MeanSquaredError(
                        neuralNetwork.Run(tuple.Item1),
                        tuple.Item2
                        );
                    cost += dC;
                }

                return (float)(cost / TrainingDataRaw.Length);
            }, -range * 0.5f, range * 0.5f);
        }

        private string GeneratePath(string append)
        {
            return $"{outputPath}Layer{layer} Row{Location.X} Col{Location.Y}{append}.png";
        }

    }
}
