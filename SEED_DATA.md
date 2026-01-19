We constructed a unified training table by using the LUCAS 2018 soil dataset as the main reference and enriching it with soil texture fractions (clay, silt, sand, coarse fragments) from LUCAS 2015 through a left join on POINT_ID. This choice preserves the full 2018 sample set while imputing texture as a quasi-static soil descriptor. In precision agriculture and management zone delineation, soil texture is commonly treated as a stable or slowly changing property, typically sampled infrequently, in contrast to more dynamic soil variables and seasonal crop indicators. For example, dynamic management zone research explicitly notes that “soil texture data are not changed or changed slowly”, whereas other agronomic variables are monitored more frequently 

Soil properties zoning of agric…

. Similarly, remote-sensing driven soil monitoring frameworks emphasize the role of soil texture as a key physical property underlying crop performance and management decisions, while highlighting that some properties (e.g., SOM) can change faster and pose greater challenges for reliable short-term prediction 

Understanding Fields by Remote …

 

Understanding Fields by Remote …

. Our output schema therefore separates identifiers and sampling context (POINT_ID, coordinates, survey date), chemical properties (pH, OC, CaCO3, nutrients, oxides) and structural properties (texture fractions), enabling subsequent modeling steps to leverage stable covariates alongside potentially time-varying soil indicators.