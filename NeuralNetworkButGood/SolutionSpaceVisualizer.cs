 using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;

namespace NeuralNetworkButGood
{
    [System.Runtime.Versioning.SupportedOSPlatform("windows")]
    internal class SolutionSpaceVisualizer
    {
        public static void Draw(int w, int h, string path, Func<int, int, Color> func)
        {
            Bitmap bmp = new Bitmap(w, h);

            for(int i = 0; i < w; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    bmp.SetPixel(i,j, func(i,j));
                }
            }

            bmp.Save(path);
        }

        public static void DrawRanges(int w, int h, string path, Vector2 Xranges, Vector2 Yranges, Func<float, float, Color> func)
        {
            Draw(w, h, path, (i, j) =>
            {
                float x = Map(i, 0, w - 1, Xranges.X, Xranges.Y);
                float y = Map(j, 0, h - 1, Yranges.X, Yranges.Y);
                return func(x, y);
            });
        }

        private static float Map(float value, float fromLow, float fromHigh, float toLow, float toHigh)
        {
            return (value - fromLow) / (fromHigh - fromLow) * (toHigh - toLow) + toLow;
        }
    }
}
