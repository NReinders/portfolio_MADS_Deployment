# Summary week 3

Experimenten met RNN-modellen op de Smartwatch Gestures Dataset

Doel

Het doel van dit experiment was om een basismodel voor gesture classification te verbeteren met behulp van recurrente neurale netwerken (RNN’s). Daarbij zijn verschillende varianten getest: een eenvoudig RNN (BaseRNN), een LSTM-model, en een hybride model met een Conv1D-laag gevolgd door een LSTM. We hebben geëvalueerd welke aanpak de beste prestaties levert en gekeken of er sprake was van overfitting.

Data en aanpak

De gebruikte dataset is de Smartwatch Gestures Dataset, bestaande uit accelerometerdata van 20 verschillende gestures. Elke sample bevat een sequentie met 3 features (x-, y- en z-acceleratie). De taak is om deze sequenties correct te classificeren in één van de 20 klassen.

Voor de experimenten is gebruikgemaakt van de mltrainer library voor training en logging. Runs zijn bijgehouden in MLflow voor systematische vergelijking. De volgende varianten zijn getest:
	1.	BaseRNN
	•	Configuratie: hidden_size=64, num_layers=1.
	•	Resultaat: accuracies rond de 6–10% (vergelijkbaar met random gokken).
	2.	LSTM
	•	Configuratie: hidden_size=128, num_layers=2, dropout=0.2.
	•	Resultaat: accuracy oplopend tot ~75–80%. Duidelijk beter dan de basismodel.
	3.	Conv1D + LSTM (hybride)
	•	Configuratie: conv_channels=16, kernel_size=3, gevolgd door een LSTM (hidden_size=128, num_layers=1, dropout=0.2).
	•	Resultaat: accuracies oplopend tot ~99% op de validatieset. Zowel training- als testloss daalden consistent, zonder groot verschil, wat wijst op goed generaliserend gedrag en weinig overfitting.

Resultaten
	•	Het toevoegen van LSTM-lagen leverde een significante verbetering ten opzichte van een eenvoudige RNN.
	•	Het combineren van een Conv1D-laag vóór de LSTM gaf het beste resultaat: korte-termijnpatronen in de sequenties werden al gefilterd door de convolutionele filters, waarna de LSTM de langere afhankelijkheden leerde.
	•	Het beste model behaalde een validatie-accuracy van 99% en een lage testloss (~0.09), met nauwelijks verschil tussen train- en testmetrics.

Conclusie

De experimenten laten zien dat het toevoegen van complexere architecturen duidelijke winst oplevert bij gesture classification. Een simpel RNN is onvoldoende krachtig, maar LSTM’s presteren aanzienlijk beter. Het beste resultaat werd behaald met een hybride Conv1D-LSTM model, dat bijna perfecte prestaties liet zien op de dataset (~99% accuracy). Er is geen sprake van sterke overfitting, wat betekent dat dit model goed generaliseert naar niet eerder geziene validatiesamples.

