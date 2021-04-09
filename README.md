# SUIPY


### Swift/Xcode

1. Open Xcode

2. Create new project > iOS > App > Next

3. Project options:

   - Name: "ImBoard"

   - Interface: SwiftUI
   - Life Cycle: SwiftUI App

4. **ContentView.swift**

```swift
import SwiftUI

struct ContentView: View {
    var body: some View {
        VStack(spacing: 20) {
            Text("I'm Board...")
                .font(.largeTitle)
            Button(action: {}) {
                Text("Predict Fun!")
            }
            Spacer()
        }
        .padding()
    }
}
```

5. New > **BoardGame.swift**

```swift
import Foundation

struct BoardGame {
    var name: String
    var time: Int
    var age: Int
    var complexity: Double
    var category: Category
    
    enum Category: String, CaseIterable {
        case abstract = "Abstract"
        case childrens = "Childrens"
        case customizable = "Customizable"
        case family = "Family"
        case party = "Party"
        case strategy = "Strategy"
        case thematic = "Thematic"
        case wargames = "Wargames"
    }
}
```

6. New > **ViewModel.swift**

```swift
import Combine

class ViewModel: ObservableObject {
    var boardGame = BoardGame(
        name: "Catan", time: 90, age: 12, complexity: 3.5, category: .strategy
    )
}
```

7. Tweak **ContentView.swift**

```swift
import SwiftUI

struct ContentView: View {
    @StateObject var viewModel = ViewModel()
    
    var body: some View {
        VStack(spacing: 20) {
            Text("I'm Board...")
                .font(.largeTitle)
            Text("\(viewModel.boardGame.name)")
            Text("\(viewModel.boardGame.time)")
            Text("\(viewModel.boardGame.age)")
            Text("\(viewModel.boardGame.complexity)")
            Text("\(viewModel.boardGame.category.rawValue)")
            Button(action: {}) {
                Text("Predict Fun!")
            }
            Spacer()
        }
        .padding()
    }
}
```

8. Editible TextField:

```swift
import SwiftUI

struct ContentView: View {
    @StateObject var viewModel = ViewModel()
    
    var body: some View {
        VStack(spacing: 20) {
            Text("I'm Board...")
                .font(.largeTitle)
            TextField("Name", text: $viewModel.boardGame.name)
            Text("\(viewModel.boardGame.time)")
            Text("\(viewModel.boardGame.age)")
            Text("\(viewModel.boardGame.category.rawValue)")
            Text("\(viewModel.boardGame.complexity)")
            Button(action: {}) {
                Text("Predict Fun!")
            }
            Spacer()
        }
        .padding()
    }
}
```

9. Add @Published to ViewModel:

```swift
import Combine

class ViewModel: ObservableObject {
    @Published var boardGame = BoardGame(
        name: "Catan", time: 90, age: 12, complexity: 3.5, category: .strategy
    )
}
```

10. Add Steppers, Sliders, and Pickers:

```swift
import SwiftUI

struct ContentView: View {
    @StateObject var viewModel = ViewModel()
    
    var body: some View {
        VStack(spacing: 20) {
            Text("I'm Board...")
                .font(.largeTitle)
            Stepper(
                "Time: \(viewModel.boardGame.time)",
                value: $viewModel.boardGame.time, in: 5...120, step: 5)
            Stepper(
                "Age: \(viewModel.boardGame.age)",
                value: $viewModel.boardGame.age, in: 4...20, step: 4)
            HStack(spacing: 10) {
                Text("Complexity")
                Slider(value: $viewModel.boardGame.complexity, in: 0...5)
            }
            Picker("Category", selection: $viewModel.boardGame.category) {
                ForEach(BoardGame.Category.allCases, id: \.self) { category in
                    Text(category.rawValue)
                }
            }
            Button(action: {}) {
                Text("Predict Fun!")
            }
            Spacer()
        }
        .padding()
    }
}
```

11. Add a dumb predict function:

```swift
class ViewModel: ObservableObject {
    @Published var boardGame = BoardGame(
        name: "Catan", time: 90, age: 12, complexity: 3.5, category: .strategy
    )
    @Published var prediction: Double?
    
    func predict() {
        prediction = 8.9
    }
}
```

12. Hook it up:

```swiftUI
import SwiftUI

struct ContentView: View {
    @StateObject var viewModel = ViewModel()
    
    var body: some View {
        VStack(spacing: 20) {
            Text("I'm Board...")
                .font(.largeTitle)
            Stepper(
                "Time: \(viewModel.boardGame.time)",
                value: $viewModel.boardGame.time, in: 5...120, step: 5)
            Stepper(
                "Age: \(viewModel.boardGame.age)",
                value: $viewModel.boardGame.age, in: 4...20, step: 4)
            HStack(spacing: 10) {
                Text("Complexity")
                Slider(value: $viewModel.boardGame.complexity, in: 0...5)
            }
            Picker("Category", selection: $viewModel.boardGame.category) {
                ForEach(BoardGame.Category.allCases, id: \.self) { category in
                    Text(category.rawValue)
                }
            }
            Button(action: viewModel.predict) {
                Text("Predict Fun!")
            }
            if let prediction = viewModel.prediction {
                Text("\(prediction)")
            }
            Spacer()
        }
        .padding()
    }
}
```



### Python

13. Create a venv

```
python -m venv .venv
```

14. Activate

```
source .venv/bin/activate
```

15. Install packages

(Note **scikit-learn==0.19.2** is the max version supported by Apple right now 😭)

```
pip install coremltools scikit-learn==0.19.2 pandas 
```

16. **01-model.py**

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# load
df = pd.read_csv('data/games.csv')

# fix
df["time"] = df["time"].apply(pd.to_numeric, errors="coerce")
df["age"] = df["age"].apply(lambda x: pd.to_numeric(x.replace("+", ""), errors="coerce"))
df = pd.concat([df, pd.get_dummies(df["category"])], axis=1)
df.columns = [c.replace("'", "").lower() for c in df.columns]
df = df.dropna()

# split
target = 'rating'
predictors = ['time', 'age', 'complexity', 'abstract', 'childrens',
    'customizable', 'family', 'party', 'strategy', 'thematic', 'wargames']
y = df[target]
X = df[predictors]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# model
model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_train, y_train), model.score(X_test, y_test))

# convert
import coremltools as ct
coreml_model = ct.converters.sklearn.convert(model, predictors, target)
coreml_model.save('models/BoardGameRegressor.mlmodel')
```

17. Run the model @ command line

```
python 01-model.py
```



### Swift/Xcode

18. Drag and drop created model into Folder Structure
19. Update the predict method in the ViewModel:

```swift

import Combine
import CoreML

class ViewModel: ObservableObject {
    @Published var boardGame = BoardGame(
        name: "Catan", time: 90, age: 12, complexity: 3.5, category: .strategy
    )
    @Published var prediction: Double?
    
    func predict() {
        let abstract = boardGame.category == .abstract ? 1.0 : 0.0
        let childrens = boardGame.category == .childrens ? 1.0 : 0.0
        let customizable = boardGame.category == .customizable ? 1.0 : 0.0
        let family = boardGame.category == .family ? 1.0 : 0.0
        let party = boardGame.category == .party ? 1.0 : 0.0
        let strategy = boardGame.category == .strategy ? 1.0 : 0.0
        let thematic = boardGame.category == .thematic ? 1.0 : 0.0
        let wargames = boardGame.category == .wargames ? 1.0 : 0.0
        do {
            let model: BoardGameRegressor = try BoardGameRegressor(configuration: .init())
            let pred = try model.prediction(
                time: Double(boardGame.time),
                age: Double(boardGame.age),
                complexity: boardGame.complexity,
                abstract: abstract,
                childrens: childrens,
                customizable: customizable,
                family: family,
                party: party,
                strategy: strategy,
                thematic: thematic,
                wargames: wargames
            )
            self.prediction = pred.rating
        } catch {
            self.prediction = nil
        }
    }
}
```

20. Make it auto-update

```swift
// BoardGame
struct BoardGame {
    var time: Int = 60
    var age: Int = 8
    var complexity: Double = 3.5
    var category: Category = .strategy
    
    enum Category: String, CaseIterable {
        case abstract = "Abstract"
        case childrens = "Childrens"
        case customizable = "Customizable"
        case family = "Family"
        case party = "Party"
        case strategy = "Strategy"
        case thematic = "Thematic"
        case wargames = "Wargames"
    }
}

// top of ViewModel
class ViewModel: ObservableObject {
    @Published var boardGame = BoardGame() {
        didSet { predict() }
    }
    @Published var prediction: Double?
  
...
  
// bottom of View
struct ContentView: View {
    @StateObject var viewModel = ViewModel()
    
    var body: some View {
        VStack(spacing: 20) {
            Text("I'm Board...")
                .font(.largeTitle)
            Stepper(
                "Time: \(viewModel.boardGame.time)",
                value: $viewModel.boardGame.time, in: 5...120, step: 5)
            Stepper(
                "Age: \(viewModel.boardGame.age)",
                value: $viewModel.boardGame.age, in: 4...20, step: 4)
            HStack(spacing: 10) {
                Text("Complexity")
                Slider(value: $viewModel.boardGame.complexity, in: 0...5)
            }
            Picker("Category", selection: $viewModel.boardGame.category) {
                ForEach(BoardGame.Category.allCases, id: \.self) { category in
                    Text(category.rawValue)
                }
            }
            if let prediction = viewModel.prediction {
                Text("\(prediction)")
            }
            Spacer()
        }
        .padding()
        .onAppear(perform: viewModel.predict)
    }
}
```



### Python

21. Update model to work with TensorFlow:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

# load
df = pd.read_csv('data/games.csv')

# fix
df["time"] = df["time"].apply(pd.to_numeric, errors="coerce")
df["age"] = df["age"].apply(lambda x: pd.to_numeric(x.replace("+", ""), errors="coerce"))
df = pd.concat([df, pd.get_dummies(df["category"])], axis=1)
df.columns = [c.replace("'", "").lower() for c in df.columns]
df = df.dropna()

# split
target = 'rating'
predictors = ['time', 'age', 'complexity', 'abstract', 'childrens',
    'customizable', 'family', 'party', 'strategy', 'thematic', 'wargames']
y = df[target]
X = df[predictors]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1])),
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(5, activation=tf.nn.relu),
    tf.keras.layers.Dense(1),
])
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(),
    loss=tf.keras.losses.mean_squared_error,
    metrics=tf.keras.metrics.mean_absolute_error
)
model.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test))

from sklearn.metrics import r2_score
r2_score(y_test, model.predict(X_test).flatten())

# convert
import coremltools as ct
coreml_model = ct.convert(model)
coreml_model.save('models/BoardGameRegressor2.mlmodel')
```

### Swift/Xcode

22. Drag & Drop and change the ViewModel to match

```
class ViewModel: ObservableObject {
    @Published var boardGame = BoardGame() {
        didSet { predict() }
    }
    @Published var prediction: Double?
    
    func predict() {
        do {
            let mlArray = try? MLMultiArray(shape: [1, 11], dataType: MLMultiArrayDataType.float32)
            mlArray![0] = NSNumber(value: boardGame.time)
            mlArray![1] = NSNumber(value: boardGame.age)
            mlArray![2] = NSNumber(value: boardGame.complexity)
            mlArray![3] = NSNumber(value: boardGame.category == .abstract ? 1.0 : 0.0)
            mlArray![4] = NSNumber(value: boardGame.category == .childrens ? 1.0 : 0.0)
            mlArray![5] = NSNumber(value: boardGame.category == .customizable ? 1.0 : 0.0)
            mlArray![6] = NSNumber(value: boardGame.category == .family ? 1.0 : 0.0)
            mlArray![7] = NSNumber(value: boardGame.category == .party ? 1.0 : 0.0)
            mlArray![8] = NSNumber(value: boardGame.category == .strategy ? 1.0 : 0.0)
            mlArray![9] = NSNumber(value: boardGame.category == .thematic ? 1.0 : 0.0)
            mlArray![9] = NSNumber(value: boardGame.category == .wargames ? 1.0 : 0.0)
            let model: BoardGameRegressor2 = try BoardGameRegressor2(configuration: .init())
            let pred = try model.prediction(input: BoardGameRegressor2Input(input_3: mlArray!))
            self.prediction = Double(truncating: pred.Identity[0])
        } catch {
            self.prediction = nil
        }
    }
}

```

