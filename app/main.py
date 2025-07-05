from fastapi import FastAPI
from contextlib import asynccontextmanager
from typing import Annotated
from decimal import Decimal
from enum import Enum
from typing import List

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlmodel import Field, Session, SQLModel, create_engine, select, Relationship, update
from sqlalchemy.orm import selectinload
from sqlalchemy import inspect  

import instructor
import google.generativeai as genai
from pydantic import BaseModel

client = instructor.from_gemini(
    client=genai.GenerativeModel(
        model_name="models/gemini-2.5-flash-lite-preview-06-17",
    )
)

class Soda(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    quantity: int | None = Field(default=None, index=True)
    price: Decimal = Field(default=0, max_digits=5, decimal_places=2)
    unitssold: list["UnitsSold"] = Relationship(back_populates='soda')

class Transaction(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    units_sold: list["UnitsSold"] = Relationship(back_populates='transaction')

class UnitsSold(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    unit_price: Decimal = Field(default=0, max_digits=5, decimal_places=2)
    quantity: int = Field(default=0)
    soda_id: int | None = Field(default=None, foreign_key="soda.id")
    soda : Soda = Relationship()
    transaction_id: int | None = Field(default=None, foreign_key="transaction.id")
    transaction: Transaction = Relationship(back_populates='units_sold')

sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)

def get_session():
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_session)]

def create_db_and_tables():
    inspector = inspect(engine)

    existing_tables = inspector.get_table_names()
    is_first_time = not existing_tables
    
    SQLModel.metadata.create_all(engine)
    
    if is_first_time:
        coke = Soda(name='Coke', quantity=100, price=Decimal(5.99))
        pepsi = Soda(name='Pepsi', quantity=10, price=Decimal(8.99))
        fanta = Soda(name='Fanta', quantity=33, price=Decimal(3.99))
        with Session(engine) as session:
            session.add(coke)
            session.add(pepsi)
            session.add(fanta)
            session.commit()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    create_db_and_tables()

    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "Hello Worlddddd"}


@app.get("/hello")
async def root_two():
    return {"message": "Hello World 2"} 

@app.get("/sodas")
async def get_sodas(session: SessionDep):
    sodas = session.exec(select(Soda)).all()
    sodas = [soda for soda in sodas]
    return sodas

class UnitsSoldRead(BaseModel):
    id: int
    unit_price: Decimal
    quantity: int
    soda_id: int
    transaction_id: int

    model_config = {
        "from_attributes": True
    }

class TransactionRead(BaseModel):
    id: int
    units_sold: List[UnitsSoldRead]
    
    model_config = {
        "from_attributes": True
    }

@app.get("/transactions", response_model=List[TransactionRead])
async def get_transactions(session: SessionDep):
    transactions = session.exec(select(Transaction).options(selectinload(Transaction.units_sold))).all()
    for transaction in transactions:
        print(transaction.units_sold)
    return transactions 


class ChatRequest(BaseModel):
    content: str


class IntentionEnum(Enum):
    buy = 'buy'
    list_sodas = 'list_sodas'
    list_transactions = 'list_transactions'
    not_possible_to_identify_intention = 'not_possible_to_identify_intention'


class SodaType(Enum):
    coke = 'Coke'
    pepsi = 'Pepsi'
    fanta = 'Fanta'


class UserTransaction(BaseModel):
    soda_type: SodaType 
    quantity: int 


class UserIntention(BaseModel):
    intention: IntentionEnum 
    transactions: List[UserTransaction] | None



def create_transaction_with_sodas(session: Session, soda_sales: list[UserTransaction]) -> Transaction:
    """
    Creates a transaction for multiple sodas sold.
    
    :param session: SQLModel Session object.
    :param soda_sales: A list of tuples [(soda_id, unit_price), ...].
    :return: The created Transaction object.
    """
    with session.begin():
        try:
            transaction = Transaction()
            
            for soda_sale in soda_sales:
                statement = select(Soda).where(Soda.name == soda_sale.soda_type.value).with_for_update()
                results = session.exec(statement)
                soda = results.one()
                if soda.quantity - soda_sale.quantity >= 0:
                    soda.quantity -= soda_sale.quantity
                    unit_sold = UnitsSold(unit_price=soda.price,
                                        soda=soda,
                                        soda_id=soda.id,
                                        quantity=soda_sale.quantity
                                        )
                    transaction.units_sold.append(unit_sold)
                    session.add(soda)
                    session.add(unit_sold)
                else:
                    raise Exception("Failed to create transaction, not enough items in stock.") 
            session.add(transaction)
            #session.refresh(transaction)
            return transaction
        except Exception as e:
            raise e
    session.refresh(transaction)
    return transaction


@app.post("/chat")
async def chat(request: ChatRequest, session: SessionDep):

    resp: UserIntention = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": f"""You are a classifier for user intent on a chatbot application.
            Classify the user intent into 4 possible cases:
            listing available sodas for sale, listing all the transactions in the database, buy something, not able to identify intention

            If there is anything weird in the user order like asking for things that dont exist, say that you cant identify the intention.
            """,
        },
        {
            "role": "user",
            "content": request.content,
        }
    ],
    response_model=UserIntention,
)

    if resp.intention == IntentionEnum.list_sodas:
        sodas = session.exec(select(Soda)).all()
        resp = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"""
                You are a chatbot for a soda vending machine.
                A user has requested a list of all the sodas offered for sale. Reply in a clear and factual message. Dont show the id of the soda.
                Here is a list of all the sodas we offer: [{sodas}]
                """,
            },
            {
                "role": "user",
                "content": request.content,
            }
        ],
        response_model=str,
    )
        return {'message': resp}
    elif resp.intention == IntentionEnum.list_transactions:
        transactions = session.exec(select(Transaction).options(selectinload(Transaction.units_sold))).all()
        transaction_data = [TransactionRead.model_validate(t).model_dump() for t in transactions]
        resp = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"""
                You are a chatbot for a soda vending machine.
                A user has requested something regarding transactions. Reply in a clear and factual message..
                Here is a list of all the transactions we have: [{transaction_data}]
                """,
            },
            {
                "role": "user",
                "content": request.content,
            }
        ],
        response_model=str,
    )
        return {'message': resp}
    elif resp.intention == IntentionEnum.buy:
        if not resp.transactions:
            raise Exception("We have failed to undestand your order, please try again.")
        transaction = create_transaction_with_sodas(session, resp.transactions)
        resp = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"""
                You are a chatbot for a soda vending machine.
                A user has requested an order and it has sucessfully been executed.
                Here are the details of the transaction: {resp.transactions}
                Tell the user a nice message.
                """,
            },
            {
                "role": "user",
                "content": request.content,
            }
        ],
        response_model=str,
    )
        return {'message': resp}
        
    elif resp.intention == IntentionEnum.not_possible_to_identify_intention:
        resp = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"""
                You are a chatbot for a soda vending machine.
                A user has requested something but we were unable to identify his intention.
                He might ask: to buy something, to list all sodas or to list all transactions 
                Try to redirect the user towards the normal workflow.
                """,
            },
            {
                "role": "user",
                "content": request.content,
            }
        ],
        response_model=str,
    )
        return {'message': resp}
        
    return {'message': resp}
