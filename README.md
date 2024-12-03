# Deep Focus Matting: Light-Weight Trimap-Free Portrait Image Matting

### A trimap free approach to portrait image matting using DeepLabv3+ with asymmetric decoders and shared encoder architecture.
 
 The process of estimating the alpha value of unknown foreground areas from photographs is known as image matting. Prior approaches required expensive auxiliary inputs or involved numerous phases that were computationally intensive. We investigate the unique roles of semantics and details in most recent portrait matting works, as well as the task breakdown into two simultaneous sub-tasks, high-level semantic segmentation and low-level details matting . For real-time portrait matting using a single input image, we come up with a light Deep Focus Matting (DFM) network. Which learns both tasks collaboratively by using a shared encoder and two separate asymmetric decoders while addressing domain-shift problem, generalization problem and localization problem. Furthermore, due to the scarcity of portrait photographs accessible for the matting task, we use composite photos for training and evaluation from the most recent works to provide a fair comparison. Earlier approaches often had a negative impact on their capacity to generalize to real-world images due to shortage of good portrait photographs. Furthermore, we provide a benchmark containing 481 foreground portrait images and 100 high-quality portrait images along with their ground truth alpha mattes. And the benchmark demonstrated similar performance with notable differences among them. Experiments on several matting benchmarks revealed that our proposed method has more room for improvement over the current state-of-the-art methods.

```see dev branch```
