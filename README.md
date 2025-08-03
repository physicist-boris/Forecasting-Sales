# Forecasting Sales

## Configuration intiale

Deux méthodes sont possibles.

Aller d'abord dans le dossier src/ (racine du projet)

```
cd src
```
### 1-Éxecution dans le powershell (Méthode 1)

```
.\initial_setup.ps1
```

### 2-Configuration manuelle (Méthode 2)

#### Création de l'environnement conda 

La commande suivante est pour créer un environnement conda avec le nom commentaires_inference.

```
conda create -n "forecasting-sales" python=3.11.0
```

#### Installation des dépendances

```
conda activate forecasting-sales
```

```
pip install -r ./env/requirements-dev.txt
```

```
pre-commit install
```
#### Autres configurations

Dans un conda PowerShell, configuré la variable d'environnement.

Ajouter le lien de la racine (dossier src/) du projet dans le PYTHONPATH
```
$env:PYTHONPATH = $pwd
```

## Configurations et commandes pour le développement

### Testing


```
python -m pytest .\tests\unit --cov=forecasting_sales -v --cov-report=html
```
```
python -m pytest .\tests\integration --cov=forecasting_sales -v --cov-report=html
```

### Pre-commit

Commande pour appliquer les pre-commit hooks sur tous les fichiers

```
pre-commit run --all-files
```

### Cyclomatic complexity

Commande permettant de calculer la complexité cyclomatique du code.

```
radon cc .
```

### Exécution des pipelines pour la prédiction (forecasting)
 
 A command line interface is made with fire to execute the nodes locally.
```
python .\run.py all
```

## Structure du code

### Noeud de processing

Le pipeline d'exécution est composé de classes appelé Noeuds. Ces class contient les méthodes transformant les données and la méthode process() est la méthode publique principale. L'idée est qu'un utilisateur ded haut niveau du pipeline n'a pas besoin de connaître les détails de l'implémentation afin de l'utiliser ([principe d'inversion des dépendances](https://fr.wikipedia.org/wiki/Inversion_des_d%C3%A9pendances)). Il est normal de séparer cette méthode dans différents méthodes (protected) et fonctions afin de faciliter l'application de tests unitaires et la lisibilité de la méthode. Si la fonction est spécifique au Noeud, il est suggéré d'utiliser une méthode protégé. Sinon, dans le cas où elle plus générales, il est suggéré de la mettre dans le module tools. La méthode process devrait être une méthode statique sans side effects (ex: saving in local files) afin qu'elle puisse facilement déplacer d'une plateform à une autre et soit facile à débugger.

```python
class MonNoeud(AbstractNode):
    """
    Mon noeud à ajouter
    """

    @log_execution
    def process(self, data: pd.DataFrame) -> Tuple[pd.DataFrame]:

        data = self._ma_fonction_1(data)
        data = self._ma_fonction_2(data)

        return data

    def _ma_fonction_1(data: pd.DataFrame) -> pd.DataFrame:
        # Mon code ici 1
        return data

    def _ma_fonction_2(data: pd.DataFrame) -> pd.DataFrame:
        # Mon code ici 2
        return data

```

### Noeuda de stockage

Les Noeuda de stockage contiennent l'implémentation de du code de persistance de la données passé entre les Noeuds. Cette approche est inspiré par le [repository pattern](https://www.cosmicpython.com/book/chapter_02_repository.html) pour abstraire les details du stockage utilisé. Il existe présentement deux types de persistences: locale et databricks. La persistence locale enregistre les données de checkpoint dans le dossier data/  et le sous-dossier associaté au Noeud. Dans databricks, mlflow est utilisé pour les objets et delta tables pour les données tabulaires. Les noeuds a utilisé (local storage nodes or databricks storage nodes) seront sélectionner au moment de l'exécution (runtime) respectivement au [strategy pattern](https://en.wikipedia.org/wiki/Strategy_pattern). Afin de pouvoir faire cette sélection, la signature la méthode pour différents types de noeuds (local or databricks) doit être la même pour les noeuds concrets. (exemple: TrainingNode).

```python
class MonNoeudLocal:
    """
    Local node storage for processing node using locla pickle files
    """
    CHECKPOINT_PATH = root_directory() / "data" / "07_mon_dossier" / "checkpoint_file.pkl"

    def __init__(self) -> None:
        self.checkpoint_predictions = NodeLocalStorage.CHECKPOINT_PATH

    def save_checkpoint(self, predictions: pd.DataFrame) -> None:
        """
        Saves the data locally
        """
        predictions.to_csv(self.checkpoint_predictions)

    def load_checkpoint(self) -> pd.DataFrame:
        """
        Load locally saved data
        """
        predictions = pd.read_csv(self.checkpoint_predictions)
        return predictions

    def load_source(self) -> Tuple[pd.DataFrame]: # Signature qui correspond à celle la méthode process du noeud associté
        """
        Load the sources for the processing node from local storage
        """
        data = pd.read_csv(self.input_data)

        return (data,) # Retourner en Tuple pour utiliser le unpack (i.e. *sources)
```

### Pipelines

Les pipelines sont définis dans le ficher run.py. Chaque méthode de la classe Pipelines correspond à une commande du CLI. La librairie Fire est utilisé afin de créer ce CLI. Lorsque sur Databricks, le package est installé sur le cluster ce qui permet d'exécuter le pipeline en appelant la méthode correspondante.

Ajouter le nouveau noeud dans le pipeline approprié (run.py)

```python
class Pipelines:
    """
    Class defining the cli interface
    """
    #  pylint: disable=too-few-public-methods
    @staticmethod
    def all(storage: StorageTypes = StorageTypes.LOCAL,
            spark_session: Optional[SparkSession] = None) -> None:
        """
        cli definition to execute node
        """
        pipeline = Pipeline(local_storage_nodes = [InferenceLocalStorage, MonNouveauNoeudStorageLocal],
                            databricks_storage_nodes = [InferenceDatabricksStorage, MonNouveauNoeudStorageDatabricks],
                            process_nodes = [InferenceNode, MonNouveauNoeud])

        NodesManager.execute_pipeline(pipeline, storage, spark_session)


    @staticmethod
    def inference(source_run_id: Optional[str] = None, storage: StorageTypes = StorageTypes.LOCAL,
                  spark_session: Optional[SparkSession] = None) -> None:
        """
        cli definition to execute node
        """
        pipeline = Pipeline(local_storage_nodes = [InferenceLocalStorage, MonNouveauNoeudStorageLocal],
                            databricks_storage_nodes = [InferenceDatabricksStorage, MonNouveauNoeudStorageDatabricks],
                            process_nodes = [InferenceNode, MonNouveauNoeud])

        NodesManager.execute_pipeline(pipeline, storage, spark_session, source_run_id)
```

### Modifications

#### Ajout d'un noeud

Pour ajouter un noeud, il faut accomplir les actions suivantes: créer un dossier dans data/, créer le noeud en héritant de la classe abstraite, créer le noeud de stockage local et créer le noeud de stockage databricks (si nécessaire). Il faut ensuite ajouter le noeud dans les pipelines approprié dans run.py.

#### Modification du code

Il est parfois nécessaire de modifier du code existant. Cela peut être fait locallement en modifiant la méthode process du noeud et/ou modifié/ajouter des méthode protégé ou fonction (dans  tools/). Lorsque sur Databricks, il est suggéré d'utiliser le noeud de stockage correspondant de databricks pour extraire la donnée. Il est possible de recréer le noeud d'exécution dans votre notebook afin d'effectuer des tests. Lorsque vous êtes satisfait du résultat, vous pouvez déplacer le code dans votre branche local et ouvrir une pull request pour intégrer les modification dans master.

### couches dans le code

Le code est divisé en quatre couche. La premmière est la couche d'exécution. C'est la couche de haut niveau pour l'utilisation des pipelines. Elle est composé de l'interface en ligne de commande (CLI) , la définition des pipelines (voir run.py), les notebooks et les outils pour exécuter ces pipelines. Elle s'intéresse principale à quoi faire (noeud à exécuter) et quelle platforme utilisée (locale ou databricks). La seconde couche la couche des noeuds de processing. À ce niveau, On s'intéresse principalement des résultats des noeuds et de leurs intrants (voir la méthode process du noeud qui nous intéresse). La troisième couche est celle contenant les détails d'implémentation d'un noeud. On s'intéresse alors au comment. Celle couche se doit d'être indépendante de la platforme utilisé. Elle est composé de méthode protégé et de fonction dans le module d'outils (/tools).  La quatrième couche est la couche de persistance composé des noeuds de stockage locale et dans Databricks storage nodes où on trouve quoi et comment stocké les informations nécessaires.

![image couches du code](./docs/couches_code.png "couches du code")