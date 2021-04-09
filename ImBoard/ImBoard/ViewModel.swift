//
//  ViewModel.swift
//  ImBoard
//
//  Created by max on 2021-04-08.
//

import Combine
import CoreML

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
