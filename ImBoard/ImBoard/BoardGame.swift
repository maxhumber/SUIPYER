//
//  BoardGame.swift
//  ImBoard
//
//  Created by max on 2021-04-08.
//

import Foundation

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
