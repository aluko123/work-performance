from sqlalchemy import Column, Integer, String, DateTime, JSON, Text, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base

class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    source_filename = Column(String, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    utterances = relationship("Utterance", back_populates="analysis")

class Utterance(Base):
    __tablename__ = "utterances"

    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id"))
    
    date = Column(String)
    timestamp = Column(String)
    speaker = Column(String, index=True)
    text = Column(Text)
    
    predictions = Column(JSON)
    aggregated_scores = Column(JSON)
    sa_labels = Column(JSON, nullable=True)
    is_indexed = Column(Boolean, default=False, nullable=False, server_default='0')

    
    analysis = relationship("Analysis", back_populates="utterances")    

    
