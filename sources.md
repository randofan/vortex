Okay, here's a consolidated list of summaries for all previously mentioned sources, explaining their relevance to your research paper, followed by a single LaTeX formatted text block with all citations.

## Source Summaries and Relevance

Here are the summaries and relevance of each source to your paper on "Precise Chronological Classification of Western Paintings (1600-1899) using Vision Transformers and Ordinal Classification."

**Core Technical Papers for Your Method:**

1.  **Dosovitskiy et al., 2020/2021 (ViT) (`Dosovitskiy20ViT` / `Dosovitskiy21`)**
    * **Summary**: This paper introduces the Vision Transformer (ViT) model, which applies a standard Transformer architecture directly to sequences of image patches for image classification tasks. It demonstrated that ViTs can achieve state-of-the-art results, particularly when pre-trained on large datasets, challenging the dominance of CNNs in computer vision.
    * **Relevance**: Critical as it describes the foundational architecture (ViT-B/16 pre-trained on ImageNet-21k) you are using as the backbone for your painting year prediction model. Its success in general visual recognition motivates its application to the specialized domain of art.

2.  **Hu et al., 2021/2022 (LoRA) (`Hu21LoRA` / `Hu22`)**
    * **Summary**: This work presents Low-Rank Adaptation (LoRA), a parameter-efficient fine-tuning technique for large pre-trained language (and vision) models. LoRA freezes the pre-trained model weights and injects trainable rank decomposition matrices into the layers of the Transformer, significantly reducing the number of trainable parameters.
    * **Relevance**: Essential to your methodology, as LoRA is used to fine-tune the pre-trained ViT for the specific task of painting year classification. This allows adaptation to the art domain with significantly fewer trainable parameters than full fine-tuning, making the process more computationally feasible. The paper's findings on rank choices guide your LoRA configuration.

3.  **Cao et al., 2020 (CORAL / Ordinal Regression) (`Cao20CORAL` / `Cao20Ordinal`)**
    * **Summary**: This paper introduces the CORAL (Consistent Rank Logits) method for ordinal regression using neural networks. It tackles ordinal classification by reformulating the K-class problem into K-1 binary classification tasks with shared weights, ensuring that the predicted probabilities are monotonically non-decreasing with rank.
    * **Relevance**: Directly relevant as it describes the CORAL framework you are implementing for your ordinal classification output layer. This method is chosen because predicting the exact year (1600-1899) is an ordinal task where the classes have a natural, strict order, and CORAL helps enforce this order.

4.  **Cheng et al., 2008 (Ordinal Regression) (`Cheng08Ordinal`)**
    * **Summary**: This paper presents an early neural network approach to ordinal regression, detailing how to adapt neural network structures and loss functions to handle problems where class labels have a natural ordering.
    * **Relevance**: Provides foundational context for using neural networks in ordinal classification tasks, supporting the choice of an ordinal approach for year prediction over standard categorical classification.

5.  **Abnar & Zuidema, 2020 (Attention Rollout) (`Abnar20AttentionRollout` / `Abnar20`)**
    * **Summary**: This paper introduces methods like "Attention Rollout" and "Attention Flow" to better quantify and visualize how attention propagates through the layers of Transformer models. Attention Rollout aggregates attention weights across layers to show the influence of input tokens on the final representation.
    * **Relevance**: Directly cited for one of the interpretability techniques you are using. Attention Rollout will help visualize which image patches (regions of the painting) the ViT model focuses on across its layers when making a year prediction.

6.  **Chefer et al., 2021 (Transformer Interpretability) (`Chefer21TransformerInterpretability` / `Chefer21`)**
    * **Summary**: This work proposes advanced interpretability methods for Transformers, including a technique based on Layer-wise Relevance Propagation (LRP), to generate relevance maps that are more class-specific and faithful than raw attention visualizations.
    * **Relevance**: Provides another key interpretability technique ("Transformer Attribution") for your research. This method will offer a more nuanced, class-specific (i.e., year-specific) understanding of which input features are most relevant for the model's chronological predictions.

7.  **Oquab et al., 2023 (DINOv2) (`Oquab23`)**
    * **Summary**: This paper presents DINOv2, a self-supervised learning method for training Vision Transformers that learn robust visual features without explicit labels. DINOv2 models, pre-trained on a large curated dataset, demonstrate strong performance on various downstream tasks and exhibit emergent properties like semantic segmentation from attention heads.
    * **Relevance**: While your current model uses a supervised pre-trained ViT, DINOv2 is relevant as it shows the power of ViTs (especially self-supervised ones) in capturing semantic visual information via attention. This supports your use of ViT attention for interpretability and could be a future direction for backbone pre-training.

**AI and Art History (General Context, Datasets, Methods):**

8.  **Cetinic & She, 2021 (Understanding and creating art with AI...) (`Cetinic21`)**
    * **Summary**: A review paper covering AI's application in both analyzing existing art (classification, retrieval, aesthetics) and creating new art. It discusses the evolution from hand-crafted features to CNNs and transfer learning, and lists various artwork datasets.
    * **Relevance**: Provides broad context on AI in art history. It mentions datasets like WikiArt and the use of CNNs for style/genre classification, which are precursors to more fine-grained temporal analysis.

9.  **Mensink & van Gemert, 2014 (The Rijksmuseum Challenge...) (`Mensink14`)**
    * **Summary**: Introduces the Rijksmuseum dataset (112k artworks, including paintings from 1500-1900) and challenges, including "Estimating Creation Year." Baselines used Fisher Vectors and SVM regression, with reported MAE around 72 years for year estimation.
    * **Relevance**: Directly relevant as it establishes a benchmark and baseline performance for year/period prediction on a historical painting dataset covering part of your target range. Useful for contextualizing your model's MAE.

10. **Strezoski & Worring, 2017/2018/2019 (OmniArt) (`Strezoski17OmniArt`, `Strezoski18`, `Strezoski19OmniArt`)**
    * **Summary**: Introduces the OmniArt method and dataset (aggregating sources like Rijksmuseum, The Met, WGA) for multi-task learning on artistic attributes including artist, type, materials, and period/year. Uses ResNet-50. The 2018 TOMM version (`Strezoski18`) reports results on the OmniArt dataset, including year prediction (regression) achieving ~70 MAE on Rijks'14. The 2019 PAMI version (`Strezoski19OmniArt`) further details learning visual representations for fine art history.
    * **Relevance**: Crucial baseline. OmniArt performs period estimation (related to your exact year task) on large-scale art datasets covering your target period. Its MAE provides a key comparison point. Your paper will compare against its methodology and results.

11. **Shen, Efros, & Aubry, 2019 (Discovering Visual Patterns...) (`Shen19`)**
    * **Summary**: Focuses on discovering near-duplicate patterns in artworks using self-supervised learning to adapt deep features (ResNet-18 conv4) for style-invariant matching, evaluated on the Brueghel dataset.
    * **Relevance**: While not directly year prediction, its method of learning dataset-specific, style-aware features from historical art could inspire techniques for feature learning sensitive to temporal stylistic changes. The Brueghel dataset is from a relevant historical period.

12. **Seguin et al., 2016 (Visual Link Retrieval...) (`Seguin16`)**
    * **Summary**: Investigates retrieving visual patterns shared by paintings using CNNs (VGG16), comparing pre-trained features to fine-tuned ones (triplet loss) on a new dataset of linked paintings from the Web Gallery of Art (WGA, 1400-1800). Found fine-tuning significantly improved performance.
    * **Relevance**: The WGA dataset covers your target period. The success of fine-tuning CNNs for art-specific visual similarity tasks is relevant. Demonstrates that features adapted to art outperform generic ones.

13. **Garcia & Vogiatzis, 2018 (How to Read Paintings: SemArt...) (`Garcia18SemArt`)**
    * **Summary**: Introduces the SemArt dataset (21k European paintings from WGA, 8th-19th C.) with images, attributes (including "timeframe" in 50-year bins), and textual artistic comments. Proposes multi-modal models for text-to-image retrieval.
    * **Relevance**: The SemArt dataset with its 50-year "timeframe" annotations for European art up to 1900 is directly relevant for temporal analysis. The multi-modal approach, while different, highlights the richness of art metadata.

14. **Garcia, Renoust, & Nakashima, 2020 (ContextNet...) (`Garcia19ContextNet` / `Garcia19UnderstandingArt`)**
    * **Summary**: Proposes ContextNet to capture contextual artistic information for painting classification and retrieval, using either multitask learning or knowledge graphs built from SemArt metadata (including "timeframe"). Showed improved classification of attributes like timeframe compared to visual-only baselines.
    * **Relevance**: Directly relevant as it tackles "timeframe" classification (50-year bins) on European paintings covering your period. Shows that incorporating relationships between artistic attributes (which co-evolve temporally) can improve period prediction.

15. **Ypsilantis et al., 2021 (The Met Dataset...) (`Ypsilantis21`)**
    * **Summary**: Introduces The Met Dataset (400k images, 224k exhibits) for instance-level artwork recognition. Training images are studio shots; query images are visitor photos. Explores various training strategies (contrastive learning, kNN classifiers).
    * **Relevance**: Provides a massive, high-quality dataset of artworks spanning diverse periods (including 1600-1899 Western art). While the paper focuses on instance recognition, the dataset itself (with its underlying museum metadata which includes dates) is a valuable resource for pre-training or augmenting training for year prediction. Findings on effective training for art data (e.g., kNN for long-tail) could be informative.

16. **Mao, Cheung, & She, 2017 (DeepArt: Learning Joint Reps...) (`Mao17`)**
    * **Summary**: Presents DeepArt, a framework to learn joint content and style representations for visual arts using a dual-path CNN (VGG-16 based) with Gram matrices for style. Introduces Art500k dataset (550k+ artworks from WGA, WikiArt, Rijksmuseum, Google Arts) with labels like artist, art movement, genre.
    * **Relevance**: Art500k is a large-scale dataset covering the target period and including Western paintings. The DeepArt framework's emphasis on learning "style" features (via Gram matrices) is relevant as style is a key indicator of period.

17. **Crowley & Zisserman, 2016 (The Art of Detection) (`Crowley16Detection`)**
    * **Summary**: Focuses on object category recognition in paintings by training classifiers on natural images and addressing domain shift. Uses the Art UK dataset (200k+ paintings). Finds ResNet architectures good for domain invariance and detectors helpful for small objects.
    * **Relevance**: The Art UK dataset is a large source of historical European paintings. Findings on robust CNNs (ResNet) for art are generally applicable.

18. **Tan, Chan, Aguirre, & Tanaka, 2016 (Ceci n'est pas une pipe...) (`Tan16`)**
    * **Summary**: Studies large-scale style, genre, and artist classification on the Wikiart dataset (80k+ paintings) using an end-to-end AlexNet-inspired CNN. Found fine-tuning an ImageNet pre-trained model performed best.
    * **Relevance**: Uses the WikiArt dataset (covering 1600-1899) and confirms the effectiveness of fine-tuned CNNs for art classification (style/genre are period-related).

19. **Chu & Wu, 2018 (Image Style Classification based on Learnt Deep Correlation Features) (`Chu18`)**
    * **Summary**: Proposes "deep correlation features" (Gram-based and others from VGG-19 feature maps) and a framework to *learn* these correlations for image style classification, primarily on oil paintings (OilPainting dataset from WikiArt). LDCF significantly outperformed GDCF.
    * **Relevance**: Focuses on extracting robust style features from paintings, covering historical styles. These advanced style features could be adapted or provide inspiration for year prediction features, as style is a strong temporal marker.

20. **Bin et al., 2024 (GalleryGPT...) (`Bin24`)**
    * **Summary**: Introduces GalleryGPT (LLaVA-based LMM) fine-tuned on a new "PaintingForm" dataset (19k famous paintings with 50k LLM-generated formal analyses focusing on visual characteristics). Aims to improve LMMs' ability to analyze artworks visually.
    * **Relevance**: Demonstrates recent LMM work on art analysis. The PaintingForm dataset, while focused on textual analysis, contains paintings from relevant periods. GalleryGPT's ability to discuss visual characteristics could be relevant if it captures period-specific visual language.

21. **Yalniz, Jégou, Chen, Paluri, & Mahajan, 2019 (Billion-scale semi-supervised...) (`Yalniz19SSL` / `Yalniz19`)**
    * **Summary**: Proposes a teacher/student semi-supervised learning pipeline to improve image classification by leveraging billions of unlabeled images.
    * **Relevance**: Offers a methodology that could be applied to your task if a large corpus of unlabeled historical paintings is available, to augment your labeled dataset and potentially improve model robustness.

22. **Karayev et al., 2014 (Recognizing Image Style) (`Karayev14`)**
    * **Summary**: Focuses on predicting image style for photographs and paintings. Introduces a "Wikipaintings" dataset (85k paintings, 25 styles, including Renaissance to modern). Found deep features (DeCAF from AlexNet) trained on ImageNet performed best for style.
    * **Relevance**: Directly addresses style classification for historical paintings covering your period, using a large dataset. The success of pre-trained CNN features for style is a key related finding. Their Wikipaintings dataset is relevant.

23. **Alayrac et al., 2022 (Flamingo...) (`Alayrac22`)**
    * **Summary**: Introduces Flamingo, a Visual Language Model (VLM) for few-shot learning on multimodal tasks by bridging pre-trained vision and language models.
    * **Relevance**: While a general VLM, its few-shot capabilities and ability to process interleaved image/text could be explored for year prediction if framed appropriately, though not its primary design. Its art-related examples show some painting understanding.

24. **Elgammal et al., 2018 (The Shape of Art History...) (`Elgammal18`)**
    * **Summary**: Investigates how CNNs classify art styles (using WikiArt) and if learned representations align with Wölfflin's art historical theories. Found CNNs learn a "smooth temporal arrangement" correlated with creation time, even without explicit date input.
    * **Relevance**: Highly relevant. Directly shows CNNs can learn temporal progression from visual style. Their methodology and finding that learned features correlate with time and art historical concepts is foundational for your work.

**Dataset Curation and Art History Context:**

25. **Getty CDWA (`GettyCDWA`)**
    * **Summary**: The Categories for the Description of Works of Art (CDWA) by Getty provides guidelines for describing artworks, including how to document creation dates.
    * **Relevance**: Informs your convention for standardizing "exact year" when dealing with date ranges or uncertain dates in your dataset construction.

26. **Joconde - French Ministry of Culture (Terms of Use) (`JocondeTerms`)**
27. **Web Gallery of Art (License Information) (`WGATerms`)**
28. **WikiArt (Terms of Use) (`WikiArtTerms`)**
29. **Art UK (Terms and Conditions/Copyright) (`ArtUKTerms`)**
30. **Google Arts & Culture (Terms of Service) (`GoogleArtsCultureTerms`)**
    * **Summary (for all 5)**: These sources provide the terms of use, copyright, and licensing information for the respective databases/collections from which you sourced images and metadata.
    * **Relevance (for all 5)**: Crucial for the "Ethical Considerations and Data Licensing" section of your dataset chapter, ensuring responsible and compliant data aggregation for your research.

31. **Burke, J. (2013). Nakedness and other peoples...**
    * **Summary**: An art historical analysis of the nude in the Italian Renaissance, considering its representation and meaning in relation to diverse cultural perspectives.
    * **Relevance**: Provides art historical context on a specific theme within a period (Renaissance) that partly overlaps with or precedes your 1600-1899 range. Useful for understanding stylistic conventions.

32. **Juan, R. (2012). The turn of the skull...**
    * **Summary**: Explores the memento mori theme in early modern art, particularly the iconography of the skull, linking it to anatomical studies like those of Vesalius.
    * **Relevance**: Offers art historical insight into symbolism and themes prevalent in the early part of your target period (early modern is c. 1500-1800), which might be reflected in visual content.

33. **Arnheim, R. (1956). Art and visual perception...**
    * **Summary**: A seminal work applying Gestalt psychology to understand how viewers perceive and interpret visual elements in art.
    * **Relevance**: Provides a theoretical framework for how visual characteristics (shape, color, composition), which your model learns, contribute to artistic expression and are perceived, which can inform discussions on interpretability.

34. **Woodall, J. (2012). Laying the table...**
    * **Summary**: An art historical examination of the conventions, techniques, and meanings embedded in still life painting.
    * **Relevance**: Still life was a prominent genre during the 1600-1899 period. Understanding its "procedures" can provide context on typical subject matter and compositional styles your model might encounter.

35. **Schreiber, G., et al. (2008). Semantic annotation...**
    * **Summary**: Describes the MultimediaN E-Culture project for semantic annotation and search in cultural heritage collections using linked data and web semantics.
    * **Relevance**: Highlights early efforts in structured data and semantic search for cultural heritage, relevant to the general goal of making art collections more accessible and analyzable through computational means.

36. **Beckett, W., & Wright, P. (1994). The story of painting.**
37. **Mishory, A. (2000). Art history: an introduction.**
    * **Summary (for both)**: These are likely general introductory texts to the history of painting or art history.
    * **Relevance (for both)**: Provide broad art historical context that can inform the introduction and discussion of stylistic periods within your 1600-1899 range.

**Foundational AI Models (General Tech):**

38. **Brown, T., et al. (2020). Language models are few-shot learners (GPT-3).**
    * **Summary**: Introduced GPT-3, demonstrating that very large language models can perform various tasks with only a few examples provided in-context (few-shot learning), without needing fine-tuning.
    * **Relevance**: While your model is vision-based and fine-tuned, GPT-3's success highlights the power of large pre-trained models. Relevant if discussing SOTA LMMs like Gemini as baselines.

39. **Gemini Team, et al. (2023). Gemini: a family of highly capable multimodal models. (`GeminiTeam23`)**
    * **Summary**: Introduces Google's Gemini family of multimodal models, capable of processing and reasoning across text, code, images, and video.
    * **Relevance**: Gemini Pro is used as a state-of-the-art LMM baseline in your experiments for zero-shot year prediction, making this paper essential for describing that baseline.

40. **Radford, A., et al. (2021). Learning transferable visual models... (CLIP). (`Radford21CLIP`)**
    * **Summary**: Presents CLIP, which learns visual representations by training to associate images with their natural language captions using a contrastive objective. CLIP models exhibit strong zero-shot transfer to various vision tasks.
    * **Relevance**: While not directly your model's architecture, CLIP is a foundational work in learning visual representations from multimodal data. It's relevant context for discussing powerful pre-trained vision models, and your ViT backbone might be conceptually related or compared to such approaches.

---

## Consolidated LaTeX Citations

```bibtex
@string(PAMI = {IEEE Trans. Pattern Anal. Mach. Intell.})
@string(IJCV = {Int. J. Comput. Vis.})
@string(CVPR= {IEEE Conf. Comput. Vis. Pattern Recog.})
@string(ICCV= {Int. Conf. Comput. Vis.})
@string(ECCV= {Eur. Conf. Comput. Vis.})
@string(NIPS= {Adv. Neural Inform. Process. Syst.})
@string(ICPR = {Int. Conf. Pattern Recog.})
@string(BMVC= {Brit. Mach. Vis. Conf.})
@string(TOG= {ACM Trans. Graph.})
@string(TIP  = {IEEE Trans. Image Process.})
@string(TVCG  = {IEEE Trans. Vis. Comput. Graph.})
@string(TMM  = {IEEE Trans. Multimedia})
@string(ACMMM= {ACM Int. Conf. Multimedia})
@string(ICME = {Int. Conf. Multimedia and Expo})
@string(ICASSP= {ICASSP})
@string(ICIP = {IEEE Int. Conf. Image Process.})
@string(ACCV  = {ACCV})
@string(ICLR = {Int. Conf. Learn. Represent.})
@string(IJCAI = {IJCAI})
@string(PR   = {Pattern Recognition})
@string(AAAI = {AAAI})
@string(CVPRW= {IEEE Conf. Comput. Vis. Pattern Recog. Worksh.})
@string(CSVT = {IEEE Trans. Circuit Syst. Video Technol.})
@string(SPL = {IEEE Sign. Process. Letters})
@string(VR   = {Vis. Res.})
@string(JOV  = {J. Vis.})
@string(TVC  = {The Vis. Comput.})
@string(JCST  = {J. Comput. Sci. Tech.})
@string(CGF  = {Comput. Graph. Forum})
@string(CVM = {Computational Visual Media})
@string(PAMI  = {IEEE TPAMI}) % Duplicate, but provided by user this way
@string(IJCV  = {IJCV}) % Duplicate
@string(CVPR  = {CVPR}) % Duplicate
@string(ICCV  = {ICCV}) % Duplicate
@string(ECCV  = {ECCV}) % Duplicate
@string(NIPS  = {NeurIPS}) % Duplicate, note NIPS vs NeurIPS
@string(ICPR  = {ICPR}) % Duplicate
@string(BMVC  = {BMVC}) % Duplicate
@string(TOG   = {ACM TOG}) % Duplicate
@string(TIP   = {IEEE TIP}) % Duplicate
@string(TVCG  = {IEEE TVCG}) % Duplicate
@string(TCSVT = {IEEE TCSVT}) % Duplicate
@string(TMM   = {IEEE TMM}) % Duplicate
@string(ACMMM = {ACM MM}) % Duplicate
@string(ICME  = {ICME}) % Duplicate
@string(ICASSP= {ICASSP}) % Duplicate
@string(ICIP  = {ICIP}) % Duplicate
@string(ACCV  = {ACCV}) % Duplicate
@string(ICLR  = {ICLR}) % Duplicate
@string(IJCAI = {IJCAI}) % Duplicate
@string(PR = {PR}) % Duplicate
@string(AAAI = {AAAI}) % Duplicate
@string(CVPRW= {CVPRW}) % Duplicate
@string(CSVT = {IEEE TCSVT}) % Duplicate
@string(EMNLPW = {Proc. Conf. Empirical Methods in Natural Language Processing: Workshops})
@string(PRL = {Pattern Recognition Letters})
@string(TNNLS = {IEEE Trans. Neural Networks Learn. Syst.})
@string(IJCNN = {Int. Joint Conf. Neural Networks})
@string(IJMMIR = {Int. J. Multimedia Inform. Retr.}) % Added for Garcia20ContextNet

@article{Cetinic21,
  author = {Eva Cetinic and James She},
  title = {Understanding and creating art with {AI}: Review and outlook},
  journal = {arXiv preprint arXiv:2102.09109},
  year = {2021},
  note = {Version: v1 [cs.CV] 18 Feb 2021}
}

@inproceedings{Mensink14,
  author = {Thomas Mensink and Jan van Gemert},
  title = {The {Rijksmuseum} {Challenge}: Museum-Centered Visual Recognition},
  booktitle = {ICMR '14: Proceedings of the International Conference on Multimedia Retrieval},
  pages = {451--454},
  year = {2014},
  publisher = {ACM}
}

@article{Strezoski17OmniArt,
  author = {Gjorgji Strezoski and Marcel Worring},
  title = {{OmniArt}: Multi-task Deep Learning for Artistic Data Analysis},
  journal = {arXiv preprint arXiv:1708.00684},
  year = {2017},
  note = {Version: v1 [cs.MM] 2 Aug 2017}
}

@article{Strezoski18,
  author  = {Gjorgji Strezoski and Marcel Worring},
  title   = {{OmniArt}: A Large-Scale Artistic Benchmark},
  journal = TMM, 
  volume  = {14},
  number  = {4},
  pages   = {88:1--88:21},
  year    = {2018},
  note    = {This is likely the TOMM reference based on user's table. Actual TOMM page numbers would be different from an arXiv version.}
}

@article{Strezoski19OmniArt,
  author    = {Gjorgji Strezoski and Marcel Worring},
  title     = {Learning Visual Representations for Fine Art History},
  journal   = PAMI,
  volume    = {41},
  number    = {9},
  pages     = {2149--2162},
  year      = {2019}
}

@inproceedings{Shen19,
  author = {Xi Shen and Alexei A. Efros and Mathieu Aubry},
  title = {Discovering Visual Patterns in Art Collections With Spatially-Consistent Feature Learning},
  booktitle = CVPR,
  pages = {9278--9287},
  year = {2019}
}

@inproceedings{Seguin16,
  author = {Benoit Seguin and Carlotta Striolo and Isabella diLenardo and Frederic Kaplan},
  title = {Visual Link Retrieval in a Database of Paintings},
  booktitle = ECCV,
  series = {Lecture Notes in Computer Science},
  volume = {9913},
  pages = {753--767},
  year = {2016},
  note = {ECCV 2016 Workshops, Part I}
}

@inproceedings{Garcia18SemArt,
  author = {Noa Garcia and George Vogiatzis},
  title = {How to Read Paintings: Semantic Art Understanding with Multi-Modal Retrieval},
  booktitle = ECCV, 
  year = {2018},
  note = {ECCV 2018 Workshops. arXiv:1810.09617v1 [cs.CV] 23 Oct 2018}
}

@article{Garcia19UnderstandingArt,
  author = {Noa Garcia and Benjamin Renoust and Yuta Nakashima},
  title = {Understanding Art through Multi-Modal Retrieval in Paintings},
  journal = {arXiv preprint arXiv:1904.10615},
  year = {2019},
  note = {v1 [cs.CV] 24 Apr 2019. Published later as Garcia20ContextNet}
}

@article{Garcia20ContextNet,
  author = {Noa Garcia and Benjamin Renoust and Yuta Nakashima},
  title = {{ContextNet}: representation and exploration for painting classification and retrieval in context},
  journal = IJMMIR,
  volume = {9},
  pages = {17--30},
  year = {2020}
}

@inproceedings{Ypsilantis21,
  author = {Nikolaos-Antonios Ypsilantis and Guangxing Han and Sarah Ibrahimi and Noa Garcia and Nanne van Noord and Giorgos Tolias},
  title = {The {Met} Dataset: Instance-level Recognition for Artworks},
  booktitle = NIPS,
  year = {2021},
  note = {Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track}
}

@inproceedings{Mao17,
  author = {Hui Mao and Ming Cheung and James She},
  title = {{DeepArt}: Learning Joint Representations of Visual Arts},
  booktitle = ACMMM,
  pages = {1183--1191},
  year = {2017}
}

@inproceedings{Crowley16Detection,
  author = {Elliot J. Crowley and Andrew Zisserman},
  title = {The Art of Detection},
  booktitle = ECCV,
  pages = {721--737},
  year = {2016}
}

@inproceedings{Tan16,
  author = {Wei Ren Tan and Chee Seng Chan and Hernán E. Aguirre and Kiyoshi Tanaka},
  title = {Ceci n'est pas une pipe: A Deep Convolutional Network for Fine-art Paintings Classification},
  booktitle = ICIP,
  pages = {3703--3707},
  year = {2016}
}

@article{Chu18,
  author = {Wei-Ta Chu and Yi-Ling Wu},
  title = {Image Style Classification based on Learnt Deep Correlation Features},
  journal = TMM,
  volume = {20},
  number = {9},
  pages = {2491--2502},
  year = {2018}
}

@inproceedings{Bin24,
  author = {Yi Bin and Wenhao Shi and Yujuan Ding and Zhiqiang Hu and Zheng Wang and Yang Yang and See-Kiong Ng and Heng Tao Shen},
  title = {{GalleryGPT}: Analyzing Paintings with Large Multimodal Models},
  booktitle = ACMMM, 
  year = {2024},
  note = {MM '24, October 28-November 1, 2024, Melbourne, VIC, Australia. arXiv:2408.00491v1 [cs.CL] 1 Aug 2024}
}

@article{Yalniz19SSL,
  author = {I. Zeki Yalniz and Hervé Jégou and Kan Chen and Manohar Paluri and Dhruv Mahajan},
  title = {Billion-scale semi-supervised learning for image classification},
  journal = {arXiv preprint arXiv:1905.00546},
  year = {2019},
  note = {Version: v1 [cs.CV] 2 May 2019}
}

@inproceedings{Karayev14,
  author = {Sergey Karayev and Matthew Trentacoste and Helen Han and Aseem Agarwala and Trevor Darrell and Aaron Hertzmann and Holger Winnemoeller},
  title = {Recognizing Image Style},
  booktitle = BMVC,
  year = {2014},
  note = {Based on arXiv:1311.3715v3 [cs.CV] 23 Sep 2014}
}

@inproceedings{Alayrac22,
  author = {Jean-Baptiste Alayrac and Jeff Donahue and Pauline Luc and Antoine Miech and Iain Barr and Yana Hasson and Karel Lenc and Arthur Mensch and Katie Millican and Malcolm Reynolds and Roman Ring and Eliza Rutherford and Serkan Cabi and Tengda Han and Zhitao Gong and Sina Samangooei and Marianne Monteiro and Jacob Menick and Sebastian Borgeaud and Andrew Brock and Aida Nematzadeh and Sahand Sharifzadeh and Mikolaj Binkowski and Ricardo Barreira and Oriol Vinyals and Andrew Zisserman and Karen Simonyan},
  title = {Flamingo: a Visual Language Model for Few-Shot Learning},
  booktitle = NIPS,
  year = {2022}
}

@inproceedings{Elgammal18,
  author = {Ahmed Elgammal and Bingchen Liu and Diana Kim and Mohamed Elhoseiny and Marian Mazzone},
  title = {The Shape of Art History in the Eyes of the Machine},
  booktitle = AAAI,
  pages = {2183--2191},
  year = {2018}
}

@inproceedings{Dosovitskiy20ViT, 
  author    = {Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
  title     = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  booktitle = ICLR,
  year      = {2021},
  note      = {arXiv preprint arXiv:2010.11929, 2020}
}

@inproceedings{Hu21LoRA,
  author    = {Edward J. Hu and Yelong Shen and Phillip Wallis and Zeyuan Allen-Zhu and Yuanzhi Li and Shean Wang and Lu Wang and Weizhu Chen},
  title     = {{LoRA}: Low-Rank Adaptation of Large Language Models},
  booktitle = ICLR,
  year      = {2022},
  note      = {arXiv preprint arXiv:2106.06823, 2021}
}

@article{Cao20Ordinal, 
  author    = {Wenzhi Cao and Vahid Mirjalili and Sebastian Raschka},
  title     = {Rank consistent ordinal regression for neural networks with application to age estimation},
  journal   = PRL,
  volume    = {140},
  pages     = {325--331},
  year      = {2020}
}

@inproceedings{Cheng08Ordinal,
  author    = {Jianlin Cheng and Guenther Chill and Yunchuan Jiang and Stefan Russell}, 
  title     = {A Neural Network Approach to Ordinal Regression},
  booktitle = IJCNN,
  pages     = {1279--1284},
  year      = {2008}
}

@inproceedings{Abnar20AttentionRollout, 
  author    = {Samira Abnar and Willem Zuidema},
  title     = {Quantifying Attention Flow in Transformers},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)},
  pages     = {4190--4197},
  year      = {2020}
}

@inproceedings{Chefer21TransformerInterpretability, 
  author    = {Hila Chefer and Shir Gur and Lior Wolf},
  title     = {Transformer Interpretability Beyond Attention Visualization},
  booktitle = CVPR,
  pages     = {782--791},
  year      = {2021}
}

@misc{Oquab23,
  author = {Maxime Oquab and Timoth{\'e}e Durand and Jakob Verbeek and Herv{\'e} J{\'e}gou and Armand Joulin},
  title  = {{DINOv2}: Learning Robust Visual Features without Supervision},
  note   = {arXiv:2304.07193},
  year   = {2023}
}

@misc{GettyCDWA,
  author    = {{Getty Vocabulary Program}},
  title     = {Categories for the Description of Works of Art ({CDWA}): Creation Date},
  howpublished = {\url{https://www.getty.edu/research/publications/electronic_publications/cdwa/20_creation.html#20B1}},
  year      = {Accessed 2024},
  note      = {Guidelines for documenting artwork creation dates.}
}

@misc{JocondeTerms,
  author    = {{Ministère de la Culture}},
  title     = {Joconde: Conditions d'utilisation},
  howpublished = {\url{https://www.culture.gouv.fr/Mentions-legales}},
  year      = {Accessed 2024},
  note      = {Terms of use for the Joconde database. Actual specific terms page for data re-use should be confirmed.}
}

@misc{WGATerms,
  author    = {{Web Gallery of Art}},
  title     = {Web Gallery of Art: License and Copyright Information},
  howpublished = {\url{https://www.wga.hu/licence.html}},
  year      = {Accessed 2024},
  note      = {License information for the Web Gallery of Art.}
}

@misc{WikiArtTerms,
  author    = {{WikiArt}},
  title     = {{WikiArt}: Terms of Use},
  howpublished = {\url{https://www.wikiart.org/en/terms-of-use}},
  year      = {Accessed 2024},
  note      = {Terms of use for WikiArt.org.}
}

@misc{ArtUKTerms,
  author    = {{Art UK}},
  title     = {{Art UK}: Terms and conditions},
  howpublished = {\url{https://artuk.org/about/terms-and-conditions}},
  year      = {Accessed 2024},
  note      = {Terms and conditions for use of Art UK content.}
}

@misc{GoogleArtsCultureTerms,
  author    = {{Google}},
  title     = {Google Arts \& Culture: Terms of Service},
  howpublished = {\url{https://artsandculture.google.com/terms}},
  year      = {Accessed 2024},
  note      = {General Google Terms of Service apply, specific permissions for image use for research should be verified.}
}

@article{Burke13Nakedness,
  author = {Jill Burke},
  title = {Nakedness and other peoples: Rethinking the italian renaissance nude},
  journal = {Art History},
  volume = {36},
  number = {4},
  pages = {714--739},
  year = {2013}
}

@article{Juan12Skull,
  author = {Rose Marie Juan},
  title = {The turn of the skull: {Andreas Vesalius} and the early modern memento mori},
  journal = {Art History},
  volume = {35},
  number = {5},
  pages = {958--975},
  year = {2012}
}

@book{Arnheim56,
  author = {Rudolf Arnheim},
  title = {Art and visual perception: A psychology of the creative eye},
  publisher = {University of California Press},
  year = {1956}
}

@article{Woodall12StillLife,
  author = {Joanna Woodall},
  title = {Laying the table: The procedures of still life},
  journal = {Art History},
  volume = {35},
  number = {5},
  pages = {976--1003},
  year = {2012}
}

@article{Schreiber08SemanticAnnotation,
  author = {Guus Schreiber and Alia Amin and Lora Aroyo and Mark van Assem and Victor de Boer and Lynda Hardman and Michiel Hildebrand and Borys Omelayenko and Jacco van Osenbruggen and Anna Tordai and others},
  title = {Semantic annotation and search of cultural-heritage collections: {The MultimediaN E-Culture} demonstrator},
  journal = {Web Semantics: Science, Services and Agents on the World Wide Web},
  volume = {6},
  number = {4},
  pages = {243--249},
  year = {2008}
}

@book{Beckett94,
  author = {Wendy Beckett and Patricia Wright},
  title = {The story of painting},
  publisher = {Dorling Kindersley London},
  year = {1994}
}

@book{Mishory00,
  author = {Alec Mishory},
  title = {Art history: an introduction},
  publisher = {Open University of Israel},
  year = {2000}
}

@inproceedings{Brown20GPT3,
  author = {Tom Brown and Benjamin Mann and Nick Ryder and Melanie Subbiah and Jared D Kaplan and Prafulla Dhariwal and Arvind Neelakantan and Pranav Shyam and Girish Sastry and Amanda Askell and others},
  title = {Language models are few-shot learners},
  booktitle = NIPS,
  volume = {33},
  pages = {1877--1901},
  year = {2020}
}

@article{GeminiTeam23,
  author = {{Gemini Team} and Rohan Anil and Sebastian Borgeaud and Jean-Baptiste Alayrac and Jiahui Yu and Radu Soricut and Johan Schalkwyk and Andrew M Dai and Anja Hauth and Katie Millican and others},
  title = {{Gemini}: a family of highly capable multimodal models},
  journal = {arXiv preprint arXiv:2312.11805},
  year = {2023}
}

@inproceedings{Radford21CLIP,
  author = {Alec Radford and Jong Wook Kim and Chris Hallacy and Aditya Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and others},
  title = {Learning transferable visual models from natural language supervision},
  booktitle = {Int. Conf. Machine Learning (ICML)},
  pages = {8748--8763},
  year = {2021},
  publisher = {PMLR}
}
```