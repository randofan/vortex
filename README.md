# Sources

## Datasets
- [The Met](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/5f93f983524def3dca464469d2cf9f3e-Paper-round2.pdf): focused on IRL dataset, so might not be as useful. Introduction speaks on art attribution and the struggle of art historians to label artworks.
- [Joconde](https://www.pop.culture.gouv.fr/): French public collections, but links are broken
- [Rijksmuseum](https://www.rijksmuseum.nl/en/collection): only ~3.5k paintings
- [WikiArt](https://github.com/lucasdavid/wikiart): retrieve WikiArt dataset
- [Web Gallery of Art](https://www.wga.hu/index1.html):
- [Google Arts & Culture](https://artsandculture.google.com/): scrape Google Arts & Culture dataset
- [Your Paintings](https://www.bbc.co.uk/yourpaintings): scrape Your Paintings dataset
- [OmniArt](https://arxiv.org/pdf/1708.00684): Combines MET, WGA, Rijks datasets
- [Art500k](https://smedia.hkust.edu.hk/james/projects/deepart_project/frp667-maoA.pdf): scrapes WikiArt, Web Gallery of Art, Rijks Museum, and Google Arts & Culture
- [Painting-91](http://www.cat.uab.cat/~joost/papers/2014MVApainting91.pdf): doesn't include year tags, so not relevant
- [BAM](https://arxiv.org/pdf/1704.08614) mostly contemporary art, so not relevant

## Related Papers
- [SemArt](https://arxiv.org/pdf/1810.09617): develops benchmark for labeling artworks with descriptions
- [State of Art](https://www.robots.ox.ac.uk/~vgg/publications/2014/Crowley14/crowley14.pdf): uses a linear SVM to classify paintings
- [DeepArt](https://smedia.hkust.edu.hk/james/projects/deepart_project/frp667-maoA.pdf): builds a CNN to match art to real-world locations
- [CognArtive](https://arxiv.org/pdf/2502.04353v1): uses ChatGPT to generate descriptions of artworks
- [Visual Link Retrieval](https://infoscience.epfl.ch/server/api/core/bitstreams/5df05d75-25f0-4da9-8474-bea0302b980a/content): tries to link components of different artworks together
- [Understanding Art](https://arxiv.org/pdf/1904.10615): uses ensemble models to intake text and images to classify painting styles
- [Rijksmuseum](https://jvgemert.github.io/pub/mensinkICMR14rijksmuseum.pdf): uses hand-made features; does year prediction; good baseline we can test against.
- [Content-Based Image Indexing of Cultural Heritage Collections](https://hal.science/hal-01164409v1/file/picard14spm.pdf): does labeling, but focuses more on visual labels; could be interesting
- [Classification of Artistic Styles Using Binarized Features Derived from a Deep Neural Network](#): has a good section on prior work which used custom features rather than DL
- [Painter Identification Using Local Features and Naive Bayes](https://scispace.com/pdf/painter-identification-using-local-features-and-naive-bayes-1dda2jyoyu.pdf): old paper which everyone references; uses simple features to identify painter
- [Unveiling the evolution of generative AI (GAI): a comprehensive and investigative analysis toward LLM models (2021–2024) and beyond](https://jesit.springeropen.com/articles/10.1186/s43067-024-00145-1): shows genAI is competitive for image recognition tasks

## Tech Papers
- [Billion-scale image classification](https://arxiv.org/pdf/1905.00546): uses KD to train a CNN
- [OmniArt](https://arxiv.org/pdf/1708.00684): Introduction addresses how we're only tackling subset of painting analysis based on digital image representation. Does year prediction as sub-component. Multi-task learning model using CNNs
- [ContextNet](https://link.springer.com/article/10.1007/s13735-019-00189-4): multi-task learning model with knowledge graph

## Art History References
- J. Burke. Nakedness and other peoples: Rethinking the italian renaissance nude. Art History, 36(4):714–739, 2013.
- R. Juan. The turn of the skull: Andreas Vesalius and the early modern memento mori. Art History, 35(5):958–975, 2012.
- Rudolf Arnheim. 1956. Art and visual perception: A psychology of the creative eye. Univ of California Press.
- J. Woodall. Laying the table: The procedures of still life. Art History, 35(5):976–1003, 2012.
- Guus Schreiber, Alia Amin, Lora Aroyo, Mark van Assem, Victor de Boer, Lynda Hardman, Michiel Hildebrand, Borys Omelayenko, Jacco van Osenbruggen, Anna Tordai, and others. 2008. Semantic annotation and search of cultural-heritage collections: The MultimediaN E-Culture demonstrator. Web Semantics: Science, Services and Agents on the World Wide Web 6, 4 (2008), 243–249
- Beckett, W., Wright, P.: The story of painting. Dorling Kindersley London (1994)
- Mishory, A.: Art history: an introduction. Open University of Israel (2000)

## Tech References
- GPT: Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.
- Gemini: Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.
- CLIP: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.
