//
//  ContentView.swift
//  ImBoard
//
//  Created by max on 2021-04-08.
//

import SwiftUI

struct ContentView: View {
    @StateObject var viewModel = ViewModel()
    
    var body: some View {
        VStack(spacing: 20) {
            Text("I'm Board...")
                .font(.largeTitle)
            Stepper(
                "Time: \(viewModel.boardGame.time)",
                value: $viewModel.boardGame.time, in: 5...120, step: 10)
            Stepper(
                "Age: \(viewModel.boardGame.age)",
                value: $viewModel.boardGame.age, in: 4...20, step: 4)
            Stepper(
                "Complexity: \(viewModel.boardGame.complexity)",
                value: $viewModel.boardGame.complexity, in: 0...5, step: 0.1)
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

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
