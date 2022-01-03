package main

import (
    "fmt"
    "log"
    "os"
    "github.com/kniren/gota/dataframe"
)

type centroid []float64

func main() {
    irisFile, err := os.Open("Iris.csv")
    if err != nil{
        log.Fatal(err)
    }
    defer irisFile.Close()

    irisDF := dataframe.ReadCSV(irisFile)
    speciesNames := []string{
            "Iris-setosa",
            "Iris-versicolor",
            "Iris-virginica",
    }

    centroid := make(map[string]centroid)

	fmt.Println(centroid)

	for _, species := range speciesNames{
		filter := dataframe.F{
			Colname: "species"
			Comparator: "=="
			Comparando: species,
		}

		summaryDF := filtered.Describe()
		var c centroid
		for _, feature := range summaryDF.Names() {
			if feature == "column" || feature == "species"{
				continue
			}
			c = append(c,summaryDF.Col(feature).Float()[0]
		)

		centroids[species] = c
		for -,species := range speciesNames{
			fmt.Printf("centroid :", species, centroid[species])
		}

		}

		}
	}
	
}