using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace NeuronalNetwork
{
    static class GetImages
    {
        private static string _subFolder= "Files";

        public static List<DigitImage> GetImagesFromFile(string location)
        {
            List<DigitImage> images = new List<DigitImage>();

            try
            {
                Console.WriteLine("\nBegin\n");
                FileStream ifsLabels = new FileStream(Path.Combine(location, _subFolder, "train-labels-idx1-ubyte.dat"),
                 FileMode.Open); // test labels
                FileStream ifsImages = new FileStream(Path.Combine(location, _subFolder, "train-images-idx3-ubyte.dat"),
                 FileMode.Open); // test images

                BinaryReader brLabels =
                 new BinaryReader(ifsLabels);
                BinaryReader brImages =
                 new BinaryReader(ifsImages);

                int magic1 = brImages.ReadInt32(); // discard
                int numImages = brImages.ReadInt32();
                int numRows = brImages.ReadInt32();
                int numCols = brImages.ReadInt32();

                int magic2 = brLabels.ReadInt32();
                int numLabels = brLabels.ReadInt32();

                byte[][] pixels = new byte[28][];
                for (int i = 0; i < pixels.Length; ++i)
                    pixels[i] = new byte[28];

                // each test image
                for (int di = 0; di < 60000; ++di)
                {
                    for (int i = 0; i < 28; ++i)
                    {
                        for (int j = 0; j < 28; ++j)
                        {
                            byte b = brImages.ReadByte();
                            pixels[i][j] = b;
                        }
                    }

                    byte lbl = brLabels.ReadByte();


                    images.Add(new DigitImage(pixels, lbl));

                } // each image

                ifsImages.Close();
                brImages.Close();
                ifsLabels.Close();
                brLabels.Close();

                Console.WriteLine("\nEnd\n");

            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                Console.ReadLine();
            }

            return images;
        }
    }
}