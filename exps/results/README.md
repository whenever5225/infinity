## Supplementary Materials: Detailed Experimental Results

The primary experimental findings are presented in the main paper's `Experimental Results` section. Comprehensive results are provided below:

### 1. Embedding Process

**Corpus**

| Dataset | dense |  |  |  | sparse |  |  |  | tensor |  |  |  |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|  | Num. | Embedding Generation Time (ms) | Data Type | Dim. | Num. | Embedding Generation Time (ms) | Data Type | Ave. Dim. | Num. | Embedding Generation Time (ms) | Data Type | Dim. |
| MSMA(en) | 8,841,823 | 70,290,130.89 | float | 1,024 | 8,841,823 | 154,599,699.00 | float | 48 | 8,841,823 | 179,928,099.64 | float | 96 |
| TREC(en) | 171,332 | 1,005,581.37 | float | 1,024 | 171,332 | 1,216,198.56 | float | 126 | 171,332 | 566,433.03 | float | 96 |
| FIQA(en) | 57,638 | 491,775.33 | float | 1,024 | 57,638 | 478,541.43 | float | 80 | 57,638 | 1,286,331.97 | float | 96 |
| TOUC(en) | 382,545 | 3,024,533.73 | float | 1,024 | 382,545 | 2,926,745.48 | float | 168 | 382,545 | 16,386,229.50 | float | 96 |
| CQAD(en) | 40,221 | 336,382.80 | float | 1,024 | 40,221 | 361,277.54 | float | 45 | 40,221 | 766,069.06 | float | 96 |
| MCCN(zh) | 935,162 | 7,353,368.04 | float | 1,024 | 935,162 | 12,741,261.54 | float | 273 | 935,162 | 57,830,873.36 | float | 64 |
| DBPE(en) | 4,635,922 | 53,701,184.23 | float | 1,024 | 4,635,922 | 65,659,245.73 | float | 45 | 4,635,922 | 66,592,478.56 | float | 96 |
| SCID(en) | 25,657 | 207,505.07 | float | 1,024 | 25,657 | 243,072.58 | float | 121 | 25,657 | 875,633.19 | float | 96 |
| SCIF(en) | 5,183 | 58,078.95 | float | 1,024 | 5,183 | 67,255.00 | float | 150 | 5,183 | 191,859.97 | float | 96 |
| MLDR(zh) | 200,000 | 34,424,884.18 | float | 1,024 | 200,000 | 6,677,859.18 | float | 1,039 | 200,000 | 82,198,483.66 | float | 64 |
| MLDR(en) | 200,000 | 23,760,586.59 | float | 1,024 | 200,000 | 5,890,243.64 | float | 877 | 200,000 | 73,604,238.16 | float | 96 | 

**Query**

| Dataset | dense |  |  |  | sparse |  |  |  | tensor |  |  |  |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|  | Num. | Embedding Generation Time (ms) | Data Type | Dim. | Num. | Embedding Generation Time (ms) | Data Type | Ave. Dim. | Num. | Embedding Generation Time (ms) | Data Type | Dim. |
| MSMA(en) | 43 | 271.50 | float | 1,024 | 43 | 273.99 | float | 7 | 43 | 369.49 | float | 96 |
| TREC(en) | 50 | 11,218.62 | float | 1,024 | 50 | 11,165.52 | float | 14 | 50 | 828.06 | float | 96 |
| FIQA(en) | 648 | 6,560.68 | float | 1,024 | 648 | 6,795.07 | float | 12 | 648 | 8,077.19 | float | 96 |
| TOUC(en) | 49 | 15,540.02 | float | 1,024 | 49 | 15,069.31 | float | 9 | 49 | 834.09 | float | 96 |
| CQAD(en) | 1,570 | 25,481.21 | float | 1,024 | 1,570 | 23,264.34 | float | 10 | 1,570 | 16,054.11 | float | 96 |
| MCCN(zh) | 339 | 4,875.48 | float | 1,024 | 339 | 4,512.46 | float | 15 | 339 | 7,879.74 | float | 64 |
| DBPE(en) | 400 | 16,281.86 | float | 1,024 | 400 | 16,239.09 | float | 7 | 400 | 5,240.93 | float | 96 |
| SCID(en) | 1,000 | 23,286.59 | float | 1,024 | 1,000 | 22,232.68 | float | 14 | 1,000 | 11,165.88 | float | 96 |
| SCIF(en) | 809 | 16,453.94 | float | 1,024 | 809 | 18,931.91 | float | 19 | 809 | 8,860.17 | float | 96 |
| MLDR(zh) | 800 | 15,143.15 | float | 1,024 | 800 | 22,864.02 | float | 13 | 800 | 22,536.86 | float | 64 |
| MLDR(en) | 800 | 19,602.23 | float | 1,024 | 800 | 123,987.21 | float | 13 | 800 | 9,004.52 | float | 96 | 


### 2. Index Construction

**Dense Vector**

| Dataset | Index Construction Time (ms) | Peak Memory Footprint (MB) | Index Size |
| ---- | ---- | ---- | ---- |
| MSMA(en) | 8,176,949.51 | 7,204 | 9.9GB |
| TREC(en) | 86,488.60 | 1,104 | 197MB |
| FIQA(en) | 22,059.68 | 400 | 67MB |
| TOUC(en) | 12,834,418.09 | 1,316 | 438MB |
| CQAD(en) | 15,612.41 | 302 | 47MB |
| MCCN(zh) | 726,385.25 | 3,329 | 1.1GB |
| DBPE(en) | 4,886,282.18 | 11,160 | 5.2GB |
| SCID(en) | 6,876.71 | 440 | 30MB |
| SCIF(en) | 1,165.20 | 90 | 6MB |
| MLDR(zh) | 123,113.31 | 1,226 | 230MB |
| MLDR(en) | 100,266.64 | 1,113 | 230MB | 

**Sparse Vector**

| Dataset | Index Construction Time (ms) | Peak Memory Footprint (MB) | Index Size |
| ---- | ---- | ---- | ---- |
| MSMA(en) | 717,577.19 | 19,353 | 7.4GB |
| TREC(en) | 10,794.62 | 1,106 | 390MB |
| FIQA(en) | 2,407.92 | 333 | 98MB |
| TOUC(en) | 31,238.75 | 2,924 | 981MB |
| CQAD(en) | 1,031.43 | 164 | 42MB |
| MCCN(zh) | 441,890.10 | 13,722 | 5.1GB |
| DBPE(en) | 136,287.04 | 11,155 | 4.3GB |
| SCID(en) | 1,722.33 | 237 | 65MB |
| SCIF(en) | 509.04 | 114 | 19MB |
| MLDR(zh) | 296,030.63 | 9,013 | 3.2GB |
| MLDR(en) | 85,602.41 | 8,197 | 3.0GB |

**Full-Text**

| Dataset | Index Construction Time (ms) | Peak Memory Footprint (MB) | Index Size |
| ---- | ---- | ---- | ---- |
| MSMA(en) | 473,321.57 | 8,136 | 1.9GB |
| TREC(en) | 23,889.79 | 946 | 129MB |
| FIQA(en) | 6,766.42 | 320 | 37MB |
| TOUC(en) | 89,633.33 | 2,055 | 412MB |
| CQAD(en) | 3,174.73 | 293 | 21MB |
| MCCN(zh) | 623,271.41 | 3,283 | 1.5GB |
| DBPE(en) | 211,804.51 | 226 | 1.2GB |
| SCID(en) | 4,952.39 | 237 | 27MB |
| SCIF(en) | 1,988.94 | 196 | 8.7MB |
| MLDR(zh) | 3,630,470.50 | 9,060 | 1.8GB |
| MLDR(en) | 333,090.87 | 7,373 | 1.7GB | 

**Tensor**

| Dataset | Index Construction Time (ms) | Peak Memory Footprint (MB) | Index Size |
| ---- | ---- | ---- | ---- |
| CQAD(en) | 335,461.44 | 11,822 | 73MB |
| MLDR(en) | 18,205,921.99 | 123,063 | 8.8GB | 

`Note`: The peak memory footprint for constructing the tensor index on MLDR(en) reaches 120GB, approaching the 125GB main memory capacity of our evaluation server.

### 3. Retrieval Performance

**Single-Path**

<u>Dense Vector:</u>

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| MSMA(en) | 0.8302471855691421 | 0.6023950354997502 | 1,042 |
| TREC(en) | 0.7506549156013494 | 0.33371925354003906 | 253 |
| FIQA(en) | 0.5316875066031886 | 0.25176869259270807 | 304 |
| TOUC(en) | 0.39010589927368267 | 0.33583446424834584 | 370 |
| MCCN(zh) | 0.5427481205901058 | 0.2524509946950127 | 1,296 |
| DBPE(en) | 0.6922264568174045 | 0.31327792868113874 | 4,873 |
| SCID(en) | 0.326374499545755 | 0.24519896507263184 | 162 |
| SCIF(en) | 0.7150427881194572 | 0.2355932436943914 | 96 |
| MLDR(zh) | 0.26223382705902654 | 0.2601909637451172 | 384 | 

| Dataset | nDCG@1 | nDCG@5 | nDCG@10 | Query Latency (ms) | QPS | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| CQAD(en) | 0.2961783439490446 | 0.3876840215721116 | 0.407590025960497 | 0.2757288088464433 | 3,402 | 184 |
| MLDR(en) | 0.37375 | 0.4682096905452836 | 0.4890266460037763 | 0.2550598978996277 | 2,854 | 378 |

<u>Sparse Vector:</u>

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| MSMA(en) | 0.6446165178085645 | 0.4220618758090707 | 497 |
| TREC(en) | 0.7814569622883276 | 0.19653797149658203 | 181 |
| FIQA(en) | 0.3745858027241684 | 0.17629239246446446 | 184 |
| TOUC(en) | 0.626018521518353 | 0.25323945648816165 | 124 |
| MCCN(zh) | 0.3542732721563407 | 0.2593204563480303 | 4,207 |
| DBPE(en) | 0.47888959564648625 | 0.3798324495107255 | 1,025 |
| SCID(en) | 0.26263727758479666 | 0.17223644256591797 | 139 |
| SCIF(en) | 0.6469613574020779 | 0.17205931454547577 | 74 |
| MLDR(zh) | 0.4054632508335321 | 0.2902477979660034 | 2,976 |


| Dataset | nDCG@1 | nDCG@5 | nDCG@10 | Query Latency (ms) | QPS | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| CQAD(en) | 0.321656050955414 | 0.3970741573501263 | 0.409989386200992 | 0.18560840825366368 | 4,674 | 102 |
| MLDR(en) | 0.505 | 0.6015827811936542 | 0.6185578365862571 | 0.2590644359588623 | 5,171 | 2,635 | 


<u>Full-Text:</u> 

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| MSMA(en) | 0.7435040438527021 | 0.47174165415209396 | 361 |
| TREC(en) | 0.838976760917619 | 0.31907081604003906 | 74 |
| FIQA(en) | 0.32829512773431974 | 0.25017034132414634 | 126 |
| TOUC(en) | 0.6503436423570825 | 0.34686497279575895 | 73 |
| MCCN(zh) | 0.22186254604676303 | 0.28856066162612903 | 635 |
| DBPE(en) | 0.5651036544380161 | 0.3541730966506729 | 282 |
| SCID(en) | 0.3103914515669131 | 0.23557305335998535 | 139 |
| SCIF(en) | 0.7035197811120181 | 0.24713386384294955 | 79 |
| MLDR(zh) | 0.41062309940695046 | 0.2932223677635193 | 628 |

| Dataset | nDCG@1 | nDCG@5 | nDCG@10 | Query Latency (ms) | QPS | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| CQAD(en) | 0.3402777777777778 | 0.4229288641585634 | 0.40089620524234 | 0.2274429722196737 | 4,684 | 107 |
| MLDR(en) | 0.5075 | 0.6200270742922134 | 0.6337588483787528 | 0.372946560382843 | 4,075 | 431 |


<u>Tensor (Brute-Force):</u> 

| Dataset | nDCG@1 | nDCG@5 | nDCG@10 | Query Latency (ms) | QPS | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| CQAD(en) | 0.32611464968152865 | 0.39406222327733575 | 0.406776515151848 | 0.5047772340713792 | 2111 | 734 |
| MLDR(en) | 0.52625 | 0.636655566115706 | 0.652147626035905 | 0.47966599464416504 | 1684 | 91,892 |

<u>Tensor (EMVB):</u> 

| Dataset | nDCG@10 | Query Latency (ms) | QPS | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- | ---- |
| CQAD(en) | 0.181466452 | 0.519629496677666 | 2,452 | 147 |
| MLDR(en) | 0.48143583533959694 | 0.46107202768325806 | 4,488 | 10,254 |

**Two-Path**

<u>Sparse+Dense:</u>

*RRF; $k_0=10$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| MSMA(en) | 0.8136074192912583 | 0.7473867993022121 | 1,361 |
| TREC(en) | 0.8030648713065267 | 0.355377197265625 | 341 |
| FIQA(en) | 0.506423435351845 | 0.313370164694482 | 270 |
| TOUC(en) | 0.5660589122936224 | 0.37338782329948583 | 437 |
| MCCN(zh) | 0.5178710599855333 | 0.3842828410052523 | 5,292 |
| DBPE(en) | 0.6740548111923965 | 0.6328706312485884 | 5,758 |
| SCID(en) | 0.3203185464330679 | 0.33627772331237793 | 175 |
| SCIF(en) | 0.7155251643704754 | 0.3109431890241728 | 92 |
| MLDR(zh) | 0.3647320620719068 | 0.5147665739059448 | 3,333 |

| Dataset | nDCG@1 | nDCG@5 | nDCG@10 | Query Latency (ms) | QPS | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| CQAD(en) | 0.321656050955414 | 0.4330043256130929 | 0.453661912381347 | 0.32689115803712493 | 2,733 | 189 |
| MLDR(en) | 0.45375 | 0.5765496189875933 | 0.5960309311267298 | 0.6430849432945251 | 1,978 | 3,103 |

*TRF; $k_0=10$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| TREC(en) | 0.8992039705550618 | 0.43182373046875 | 1,148 |
| FIQA(en) | 0.5101732166751165 | 0.3402792948391176 | 2,750 |
| TOUC(en) | 0.5963351469160038 | 0.4744432410415338 | 715 |
| MCCN(zh) | 0.6215496614404877 | 0.5198006919361805 | 10,718 |
| DBPE(en) | 0.7111485574892028 | 0.7642862373041647 | 5,969 |
| SCID(en) | 0.35316495755753613 | 0.35440897941589355 | 1,175 |
| SCIF(en) | 0.7463409601024527 | 0.345667597406936 | 745 |
| MLDR(zh) | 0.4478959479741837 | 0.7746070623397827 | 24,919 |

| Dataset | nDCG@10 | Query Latency (ms) | QPS | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- | ---- |
| CQAD(en) | 0.46308835976949075 | 0.387009086122938 | 1,842 | 517 |
| MLDR(en) | 0.6656489301227464 | 0.7157295942306519 | 1,787 | 16,507 |

*RRF; $k_0=100$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| CQAD(en) | 0.4727491475232798 | 0.3490432812150117 | 190 |
| MLDR(en) | 0.5981491899418312 | 0.6737586855888367 | 3,114 |

*RRF; $k_0=1000$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| CQAD(en) | 0.4794865494321084 | 0.3456504481613256 | 193 |
| MLDR(en) | 0.5964799200855978 | 0.6735119223594666 | 3,115 |

*TRF; $k_0=100$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| CQAD(en) | 0.434603564831607 | 0.3911443576691257 | 803 |
| MLDR(en) | 0.6624085402538336 | 0.7609373331069946 | 54,176 |

*TRF; $k_0=1000$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| CQAD(en) | 0.413203016 | 0.7576880181670949 | 833 |
| MLDR(en) | 0.6569404640325925 | 0.8133861422538757 | 85,962 |


*WS; $k_0=10$*

| Dataset | nDCG@10 | Query Latency (ms) | QPS | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- | ---- |
| CQAD(en) | 0.422798718 | 0.2982555680973515 | 2,094 | 156 |
| MLDR(en) | 0.6203584407417759 | 0.5017656087875366 | 1,946 | 3,013 |


*WS; $k_0=100$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| CQAD(en) | 0.483881237 | 0.33350977928015835 | 158 |
| MLDR(en) | 0.6209414045357579 | 0.6072676181793213 | 3,059 |


*WS; $k_0=1000$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| CQAD(en) | 0.505446045 | 0.33403308528244113 | 160 |
| MLDR(en) | 0.6257059367380701 | 0.7579940557479858 | 3,062 |


<u>Sparse+Full-Text:</u>

*RRF; $k_0=10$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| MSMA(en) | 0.7344625480654439 | 0.5851346392964207 | 702 |
| TREC(en) | 0.8351681423478161 | 0.38321495056152344 | 194 |
| FIQA(en) | 0.393256529512051 | 0.3343632648711314 | 222 |
| TOUC(en) | 0.6901024994124113 | 0.40561325696049905 | 142 |
| MCCN(zh) | 0.32824510155622844 | 0.522000408903177 | 4,666 |
| DBPE(en) | 0.6084385916643597 | 0.5502297485155518 | 1,185 |
| SCID(en) | 0.31233467420192357 | 0.26247382164001465 | 162 |
| SCIF(en) | 0.6999138231304799 | 0.3299384207420246 | 94 |
| MLDR(zh) | 0.45475909877442566 | 0.4786732792854309 | 3,419 |

| Dataset | nDCG@1 | nDCG@5 | nDCG@10 | Query Latency (ms) | QPS | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| CQAD(en) | 0.3375796178343949 | 0.4301003786988486 | 0.444912382346468 | 0.27047949991408426 | 2,749 | 120 |
| MLDR(en) | 0.555 | 0.659610160230573 | 0.6753900228624247 | 0.4964590072631836 | 2,849 | 2,952 |

*TRF; $k_0=10$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| TREC(en) | 0.9113161229956407 | 0.4620933532714844 | 1,025 |
| FIQA(en) | 0.45689677876777374 | 0.3068012092374579 | 2,844 |
| TOUC(en) | 0.6771018117871307 | 0.3910745893205915 | 453 |
| MCCN(zh) | 0.4376683487365836 | 0.5689262992575651 | 10,124 |
| DBPE(en) | 0.6776628498194449 | 0.5869099619046524 | 1,394 |
| SCID(en) | 0.3431805735580495 | 0.3194088935852051 | 1,240 |
| SCIF(en) | 0.7472775127297523 | 0.3033118596261853 | 770 |
| MLDR(zh) | 0.49451330074898786 | 0.5832087993621826 | 25,075 |

| Dataset | nDCG@10 | Query Latency (ms) | QPS | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- | ---- |
| CQAD(en) | 0.44853700474792313 | 0.331427489116693 | 1,790 | 516 |
| MLDR(en) | 0.68992039928432 | 0.6536862254142761 | 1,740 | 15,253 |


*RRF; $k_0=100$*

| Dataset | nDCG@k | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| CQAD(en) | 0.4345210695464002 | 0.31769518639631333 | 121 |
| MLDR(en) | 0.6769675188735291 | 0.5860471725463867 | 3,008 |

*RRF; $k_0=1000$*

| Dataset | nDCG@k | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| CQAD(en) | 0.4333454323696051 | 0.33259574015429066 | 125 |
| MLDR(en) | 0.6763017209596204 | 0.6324231624603271 | 3,010 |


*WS; $k_0=10$*

| Dataset | nDCG@10 | Query Latency (ms) | QPS | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- | ---- |
| CQAD(en) | 0.421438648 | 0.31088005964923054 | 2,752 | 120 |
| MLDR(en) | 0.6565017369083727 | 0.4511776566505432 | 2,715 | 2,881 |


<u>Dense+Full-Text:</u>

*RRF; $k_0=10$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| MSMA(en) | 0.8164070112093362 | 0.6672393443972565 | 1,012 |
| TREC(en) | 0.8318157605020058 | 0.5253171920776367 | 246 |
| FIQA(en) | 0.47055787677124844 | 0.36739818886298997 | 191 |
| TOUC(en) | 0.6037135259680207 | 0.3992440749187859 | 350 |
| MCCN(zh) | 0.4400967966996403 | 0.6477519492226343 | 1,700 |
| DBPE(en) | 0.668134161082723 | 0.5477449878635448 | 5,016 |
| SCID(en) | 0.3473564128946232 | 0.3429858684539795 | 139 |
| SCIF(en) | 0.748146365389946 | 0.35208340052337234 | 82 |
| MLDR(zh) | 0.41062421813740907 | 0.5682927370071411 | 852 |

| Dataset | nDCG@1 | nDCG@5 | nDCG@10 | Query Latency (ms) | QPS | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| CQAD(en) | 0.35159235668789807 | 0.4463685235718911 | 0.462643659962836 | 0.378576205794219 | 2,059 | 219 |
| MLDR(en) | 0.52125 | 0.6337060997226025 | 0.645809935557573 | 0.5428171157836914 | 1,772 | 677 |

*TRF; $k_0=10$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| TREC(en) | 0.9202405252160172 | 0.592808723449707 | 1,039 |
| FIQA(en) | 0.5075500313745738 | 0.37403517681769083 | 2,757 |
| TOUC(en) | 0.6178205897143665 | 0.45318019633390466 | 624 |
| MCCN(zh) | 0.6244572498731534 | 0.6941584326901995 | 6,705 |
| DBPE(en) | 0.7222301917161004 | 0.6317132545487528 | 5,223 |
| SCID(en) | 0.35379179186512105 | 0.36902403831481934 | 1,111 |
| SCIF(en) | 0.760638598503032 | 0.3773516851894699 | 736 |
| MLDR(zh) | 0.47082670053220654 | 0.8115369081497192 | 21,432 |

| Dataset | nDCG@10 | Query Latency (ms) | QPS | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- | ---- |
| CQAD(en) | 0.4645431212143255 | 0.40732705669038616 | 1,326 | 519 |
| MLDR(en) | 0.6795227789852534 | 0.7343509793281555 | 1,635 | 12,927 |

*RRF; $k_0=100$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| CQAD(en) | 0.47529766241981114 | 0.38390053305656285 | 220 |
| MLDR(en) | 0.6360517754948745 | 0.6878301501274109 | 691 |

*RRF; $k_0=1000$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| CQAD(en) | 0.4844110077434764 | 0.4050746844832305 | 222 |
| MLDR(en) | 0.6361943387654986 | 0.6995967030525208 | 712 |

*WS; $k_0=10$*

| Dataset | nDCG@10 | Query Latency (ms) | QPS | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- | ---- |
| CQAD(en) | 0.430072702 | 0.3477300048633746 | 2,234 | 124 |
| MLDR(en) | 0.6381917184342016 | 0.5358591675758362 | 2,051 | 661 |

<u>Sparse+Tensor:</u>

*Tensor(Brute-Force); RRF; $k_0=10$*

| Dataset | nDCG@1 | nDCG@5 | nDCG@10 | Query Latency (ms) | QPS | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| CQAD(en) | 0.34203821656050953 | 0.42915278423276365 | 0.4441990611587147 | 0.5225492890473384 | 2,127 | 787 |
| MLDR(en) | 0.54375 | 0.6744265944943841 | 0.6895544173067344 | 0.7102817296981812 | 1,610 | 94,360 |

*Tensor(EMVB); RRF; $k_0=10$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| CQAD(en) | 0.39068662801767 | 0.641380753486779 | 260 |
| MLDR(en) | 0.6836822899834694 | 0.7336601614952087 | 13,035 |

*Tensor(Brute-Force); RRF; $k_0=100$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| CQAD(en) | 0.4366553324571527 | 0.539855896287663 | 789 |
| MLDR(en) | 0.6784478461139335 | 0.7261312007904053 | 94,395 |

*Tensor(Brute-Force); RRF; $k_0=1000$*

| nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- |
| 0.437705874919569 | 0.6106072930014057 | 791 |
| 0.6794272123824825 | 0.7260486483573914 | 94,400 |

<u>Dense+Tensor:</u>

*Tensor(Brute-Force); RRF; $k_0=10$*

| Dataset | nDCG@1 | nDCG@5 | nDCG@10 | Query Latency (ms) | QPS | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| CQAD(en) | 0.3235668789808917 | 0.43374661872002285 | 0.4501916912709554 | 0.7508416084726904 | 1,395 | 794 |
| MLDR(en) | 0.48625 | 0.6254209542093676 | 0.6402971089453392 | 0.8781886100769043 | 1,540 | 92,120 |

*Tensor(EMVB); RRF; $k_0=10$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| CQAD(en) | 0.378913882 | 0.7667538466726899 | 533 |
| MLDR(en) | 0.5991016896169015 | 0.9107819199562073 | 10,876 |

*Tensor(Brute-Force); RRF; $k_0=100$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| CQAD(en) | 0.4775988731157805 | 0.7760591567701595 | 795 |
| MLDR(en) | 0.6222049666642541 | 0.8766488528251648 | 92,139 |

*Tensor(Brute-Force); RRF; $k_0=1000$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| CQAD(en) | 0.48579012353131124 | 0.783179975618982 | 798 |
| MLDR(en) | 0.6221814881270317 | 0.8864215016365051 | 92,140 |

<u>Full-Text+Tensor:</u>

*Tensor(Brute-Force); RRF; $k_0=10$*

| Dataset | nDCG@1 | nDCG@5 | nDCG@10 | Query Latency (ms) | QPS | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| CQAD(en) | 0.354140127388535 | 0.4344044391055907 | 0.4461032542454914 | 0.545621525709796 | 1,925 | 822 |
| MLDR(en) | 0.56625 | 0.6844894381239218 | 0.7004227285905867 | 0.7538807392120361 | 2,309 | 92,013 |

*Tensor(EMVB); RRF; $k_0=10$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| CQAD(en) | 0.390081644 | 0.689250800260313 | 173 |
| MLDR(en) | 0.7017187129152905 | 0.7521626353263855 | 10,709 |

*Tensor(Brute-Force); RRF; $k_0=100$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| CQAD(en) | 0.440325120801105 | 0.5677027307498227 | 832 |
| MLDR(en) | 0.6867122812355915 | 0.7595255970954895 | 92,030 |

*Tensor(Brute-Force); RRF; $k_0=1000$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| CQAD(en) | 0.4395076733550783 | 0.6301439491806516 | 846 |
| MLDR(en) | 0.6858918602063517 | 0.7607510685920715 | 92,035 |

**Three-Path**

<u>Sparse+Dense+Full-Text:</u>

*RRF; $k_0=10$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| MSMA(en) | 0.8189457866368675 | 0.7496878158214481 | 1,515 |
| TREC(en) | 0.8649501232115431 | 0.7003974914550781 | 367 |
| FIQA(en) | 0.48986483495258515 | 0.39338050623089255 | 288 |
| TOUC(en) | 0.6544591315746802 | 0.4740977773861009 | 456 |
| MCCN(zh) | 0.47353735138066727 | 0.6831144682435197 | 5,736 |
| DBPE(en) | 0.6922366055286493 | 0.6676251168424618 | 5,919 |
| SCID(en) | 0.3448333358422739 | 0.3557851314544678 | 189 |
| SCIF(en) | 0.7386728624678759 | 0.353514986923947 | 165 |
| MLDR(zh) | 0.4506020284379539 | 0.7514846324920654 | 3,642 |

| Dataset | nDCG@1 | nDCG@5 | nDCG@10 | Query Latency (ms) | QPS | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| CQAD(en) | 0.37515923566878984 | 0.46924561747167304 | 0.486054085728164 | 0.38510462281050956 | 1,563 | 263 |
| MLDR(en) | 0.5525 | 0.6600196545692857 | 0.6761128917092672 | 0.6516137719154358 | 2,076 | 3,141 |

*TRF; $k_0=10$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| TREC(en) | 0.9238809332034128 | 0.7436084747314453 | 1,211 |
| FIQA(en) | 0.4963547991967436 | 0.4134967869369562 | 3,103 |
| TOUC(en) | 0.6270431368946908 | 0.5184630958401427 | 780 |
| MCCN(zh) | 0.6238958959599793 | 0.7622109909372324 | 12,902 |
| DBPE(en) | 0.7125199838681795 | 0.790623066512114 | 6,193 |
| SCID(en) | 0.3531999987366033 | 0.38033580780029297 | 1,419 |
| SCIF(en) | 0.7622393644202485 | 0.3901697473551799 | 754 |
| MLDR(zh) | 0.4964732748881091 | 0.8523255586624146 | 31,588 |


| Dataset | nDCG@10 | Query Latency (ms) | QPS | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- | ---- |
| CQAD(en) | 0.4650422916430904 | 0.5743468642994097 | 1,211 | 594 |
| MLDR(en) | 0.6926869892363571 | 0.873626172542572 | 1,204 | 19,542 |


*RRF; $k_0=100$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| CQAD(en) | 0.4840385257499768 | 0.39395982292807025 | 265 |
| MLDR(en) | 0.6686786483400031 | 0.7709965109825134 | 3,197 |

*RRF; $k_0=1000$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| CQAD(en) | 0.49116832057144616 | 0.41374841313453237 | 268 |
| MLDR(en) | 0.6753149512119424 | 0.7828214764595032 | 3,199 |

*WS; $k_0=10$*

| Dataset | nDCG@10 | Query Latency (ms) | QPS | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- | ---- |
| CQAD(en) | 0.465234924 | 0.3958489484847731 | 1,824 | 168 |
| MLDR(en) | 0.6689073355071354 | 0.5952996015548706 | 1,640 | 3,117 |

<u>Sparse+Dense+Tensor:</u>

*Tensor(Brute-Force); RRF; $k_0=10$*

| Dataset | nDCG@1 | nDCG@5 | nDCG@10 | Query Latency (ms) | QPS | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| CQAD(en) | 0.37515923566878984 | 0.47129465059850556 | 0.48566736044435627 | 0.7572309226746772 | 1,523 | 900 |
| MLDR(en) | 0.5525 | 0.6656199374543115 | 0.6858023683001846 | 0.8938091993331909 | 1,809 | 94,573 |

*Tensor(EMVB); RRF; $k_0=10$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| CQAD(en) | 0.446441028 | 0.8230760598638255 | 257 |
| MLDR(en) | 0.6812116209402876 | 0.8529180288314819 | 13,587 |


*Tensor(Brute-Force); RRF; $k_0=100$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| CQAD(en) | 0.4885807876273929 | 0.7620118985510176 | 910 |
| MLDR(en) | 0.6629509049999668 | 0.909000039100647 | 94,617 |

*Tensor(Brute-Force); RRF; $k_0=1000$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| CQAD(en) | 0.4941697528453086 | 0.8099202137843818 | 915 |
| MLDR(en) | 0.6642196442235221 | 0.9133359789848328 | 94,625 |

<u>Sparse+Full-Text+Tensor:</u>

*Tensor(Brute-Force); RRF; $k_0=10$*

| Dataset | nDCG@1 | nDCG@5 | nDCG@10 | Query Latency (ms) | QPS | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| CQAD(en) | 0.37707006369426754 | 0.4584121428471271 | 0.47228249478168083 | 0.617840487486238 | 1,688 | 868 |
| MLDR(en) | 0.59125 | 0.7046717171118317 | 0.7205534688289787 | 0.7590270042419434 | 1,865 | 94,464 |

*Tensor(EMVB); RRF; $k_0=10$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| CQAD(en) | 0.445552304 | 0.7146803436765246 | 214 |
| MLDR(en) | 0.7353885734164681 | 0.7363241910934448 | 13,587 |

*Tensor(Brute-Force); RRF; $k_0=100$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| CQAD(en) | 0.46443191807288203 | 0.6485498634872923 | 875 |
| MLDR(en) | 0.7068412194603146 | 0.7584688067436218 | 95,098 |

*Tensor(Brute-Force); RRF; $k_0=1000$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| CQAD(en) | 0.46605440416158983 | 0.7347582252162277 | 882 |
| MLDR(en) | 0.7118716437094887 | 0.7906970381736755 | 95,235 |

<u>Dense+Full-Text+Tensor:</u>

*Tensor(Brute-Force); RRF; $k_0=10$*

| Dataset | nDCG@1 | nDCG@5 | nDCG@10 | Query Latency (ms) | QPS | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| CQAD(en) | 0.38598726114649684 | 0.47508582200412064 | 0.48869351165780406 | 0.780812949891303 | 1,411 | 869 |
| MLDR(en) | 0.58625 | 0.6924507432763382 | 0.7087652071748125 | 0.8842051029205322 | 1,032 | 92,252 |


*Tensor(EMVB); RRF; $k_0=10$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| CQAD(en) | 0.459325273 | 0.8512366349530068 | 226 |
| MLDR(en) | 0.7068257071250263 | 0.9531968832015991 | 11,199 |

*Tensor(Brute-Force); RRF; $k_0=100$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| CQAD(en) | 0.49018128233726227 | 0.7806562314367598 | 879 |
| MLDR(en) | 0.6797789041782765 | 0.8975699543952942 | 92,259 |

*Tensor(Brute-Force); RRF; $k_0=1000$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| CQAD(en) | 0.495920356650334 | 0.8325509964280827 | 883 |
| MLDR(en) | 0.6842344984705959 | 0.9351992607116699 | 92,377 |


**Four-Path**

<u>Sparse+Dense+Full-Text+Tensor:</u>

*Tensor(Brute-Force); RRF; $k_0=10$*

| Dataset | nDCG@1 | nDCG@5 | nDCG@10 | Query Latency (ms) | QPS | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| CQAD(en) | 0.3974522292993631 | 0.48845720092183986 | 0.50204530785982 | 0.8480998361186617 | 1,425 | 913 |
| MLDR(en) | 0.5925 | 0.6981623984226689 | 0.7174141482280791 | 0.9833890199661255 | 1,391 | 94,698 |

*Tensor(EMVB); RRF; $k_0=10$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| CQAD(en) | 0.483511139 | 0.8867259238176285 | 270 |
| MLDR(en) | 0.72765602245088 | 0.9545302391052246 | 13,895 |

*Tensor(Brute-Force); RRF; $k_0=100$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| CQAD(en) | 0.4992382553083954 | 0.8713477736066102 | 922 |
| MLDR(en) | 0.699464883751146 | 1.0220322012901306 | 94,757 |

*Tensor(Brute-Force); RRF; $k_0=1000$*

| Dataset | nDCG@10 | Query Latency (ms) | Peak Memory Footprint (MB) |
| ---- | ---- | ---- | ---- |
| CQAD(en) | 0.5041841493750471 | 0.8878952378679992 | 936 |
| MLDR(en) | 0.7002872365388975 | 0.998649001121521 | 95,237 |